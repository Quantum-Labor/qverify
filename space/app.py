"""QVerify HuggingFace Space - verifier-only demo, runs on CPU Basic.

This Space exposes the QVerify *verifier* component: a CNF is fed
straight to Grover's search, on either the PennyLane simulator (local
to this Space's CPU container) or IBM Heron r2 hardware (delegated to
IBM Cloud over the network, account credentials read from Space
Secrets).

The simulator path is synchronous and returns in milliseconds. The
IBM hardware path takes 2-20 minutes wall-clock (queue + transpile +
execution + result fetch), well past Gradio's WebSocket timeout on
HF Spaces' free tier. To keep the UI responsive across that window,
the IBM verifier is implemented as a generator that yields a
"submitted" update with the IBM job_id immediately, polls the job
every 8 seconds, and yields progress + the final result. A separate
"recover by job ID" panel is provided as a fallback when the live
stream disconnects mid-run.

The translator (Gemma 4 E4B with grammar-constrained generation)
is intentionally not loaded here: it needs a GPU, which CPU Basic
does not provide. The full pipeline runs locally from the GitHub
repository, see https://github.com/Quantum-Labor/qverify.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from typing import Literal as _Literal

import gradio as gr

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.utils.ibm_client import IBMRuntimeClient
from qverify.verifier import verify
from qverify.verifier._universe import Universe
from qverify.verifier.backends import PennyLaneBackend
from qverify.verifier.classical_check import satisfies
from qverify.verifier.encoding import AtomEncoder
from qverify.verifier.grounding import ground_cnf
from qverify.verifier.grover import (
    MAX_VARIABLES,
    TOP_MEASUREMENTS_KEEP,
    optimal_iterations,
)
from qverify.verifier.grover_circuit import build_grover_qiskit_circuit
from qverify.verifier.types import CounterModel, VerificationResult

VerifyMode = _Literal["entailment", "consistency"]

IBM_CONSOLE_BASE = "https://quantum.cloud.ibm.com/workloads?search="

# Polling cadence for IBM job status. 8 s is conservative — IBM's
# backend-status endpoints rate-limit at ~1 req/s but the polling adds
# little value below ~5 s anyway because Heron jobs typically queue
# for tens of seconds at minimum.
POLL_INTERVAL_SECONDS = 8

# Hard ceiling on how long the live UI flow waits for an IBM job.
# Past this, yield a "timeout" message and tell the user to use the
# fallback panel. 600 s = 10 minutes, which covers a typical free-tier
# queue + execution window.
LIVE_POLL_TIMEOUT_SECONDS = 600

# IBM job statuses that mean "the job is finished, result is ready
# (success or failure)". Anything else is in-flight.
IBM_TERMINAL_STATUSES = frozenset({"DONE", "ERROR", "CANCELLED"})


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
    universe_str = ", ".join(ex.universe.constants) if ex.universe.constants else "(empty)"
    return _format_cnf(ex.cnf), universe_str, ex.description


# --------------------------------------------------------------------------
# Simulator path - synchronous, fast
# --------------------------------------------------------------------------


def _coerce_mode(mode: str) -> VerifyMode:
    """Narrow a Gradio-supplied string into the Literal the verifier expects."""
    if mode not in ("entailment", "consistency"):
        raise ValueError(f"unknown mode {mode!r}")
    return cast(VerifyMode, mode)


def _format_counter_model(result: VerificationResult, mode: VerifyMode) -> Any:
    """Project a VerificationResult counter_model into the JSON output shape."""
    if result.counter_model is not None:
        return result.counter_model.assignment
    if mode == "consistency":
        return (
            "(consistency mode: no satisfying assignment is surfaced; "
            "this is expected when the formula is consistent because the "
            "verifier does not display a model in this mode, and when "
            "inconsistent because UNSAT means there is none)"
        )
    return "no counter-model found (formula is entailed)"


def verify_on_simulator(label: str, mode: str) -> dict[str, Any]:
    """Run Grover on the PennyLane simulator and return the result dict."""
    if label not in EXAMPLES:
        return {"error": f"unknown example: {label}"}
    try:
        narrow_mode = _coerce_mode(mode)
    except ValueError as exc:
        return {"error": str(exc)}

    ex = EXAMPLES[label]
    grounded = ground_cnf(ex.cnf, ex.universe)

    start = time.monotonic()
    result = verify(grounded, backend=PennyLaneBackend(), mode=narrow_mode)
    wall = round(time.monotonic() - start, 1)

    return {
        "status": "completed",
        "example": label,
        "mode": narrow_mode,
        "backend": result.backend_name,
        "contradiction_found": result.contradiction_found,
        "counter_model": _format_counter_model(result, narrow_mode),
        "n_variables": result.n_variables,
        "n_clauses": result.n_clauses,
        "n_grover_iterations": result.n_grover_iterations,
        "shots": result.shots,
        "wall_clock_seconds": wall,
    }


# --------------------------------------------------------------------------
# IBM hardware path - submit, poll, decode
# --------------------------------------------------------------------------


def _build_runtime_client() -> IBMRuntimeClient:
    """Construct an IBMRuntimeClient from the Space's environment Secrets."""
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "")
    return IBMRuntimeClient(token=token, instance=instance)


@dataclass(frozen=True)
class _PreparedJob:
    """Everything we need to decode an IBM job result into a VerificationResult."""

    cnf: CNF  # the grounded propositional CNF
    encoder: AtomEncoder
    n_qubits: int
    n_iterations: int
    n_clauses: int
    shots: int


def _prepare_and_submit(label: str, shots: int = 1024) -> tuple[Any, str, str, _PreparedJob]:
    """Ground, encode, transpile, and submit the example. Return immediately
    with the live qiskit Job, its job_id, the backend name, and a
    :class:`_PreparedJob` carrying the metadata needed to decode the result
    once it lands.
    """
    ex = EXAMPLES[label]
    grounded = ground_cnf(ex.cnf, ex.universe)

    encoder = AtomEncoder(grounded)
    n_qubits = encoder.n_qubits
    n_clauses = len(grounded.clauses)
    if n_qubits > MAX_VARIABLES:
        raise ValueError(
            f"CNF has {n_qubits} variables; the Space accepts at most "
            f"{MAX_VARIABLES}. Use a smaller example."
        )
    encoded_clauses = encoder.encode_clauses()
    n_iterations = optimal_iterations(n_qubits)

    circuit = build_grover_qiskit_circuit(encoded_clauses, n_qubits, n_iterations)

    # Mirror what qverify.utils.ibm_client.IBMRuntimeClient.run does, but
    # split the blocking job.result() out of this submit phase so the UI
    # can poll between submission and completion.
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    client = _build_runtime_client()
    service = client.get_service()
    backend_name = client.least_busy_heron(min_qubits=5)
    backend = service.backend(backend_name)

    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = pass_manager.run(circuit)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    job_id = str(job.job_id())

    prepared = _PreparedJob(
        cnf=grounded,
        encoder=encoder,
        n_qubits=n_qubits,
        n_iterations=n_iterations,
        n_clauses=n_clauses,
        shots=shots,
    )
    return job, job_id, str(backend_name), prepared


def _decode_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    """Reverse Qiskit's little-endian-by-classical-bit ordering so qubit 0 is leftmost."""
    return {bs[::-1]: int(c) for bs, c in raw_counts.items()}


def _build_verification_result(
    counts: dict[str, int],
    prepared: _PreparedJob,
    backend_name: str,
    mode: VerifyMode,
) -> VerificationResult:
    """Run the same classical post-check the simulator path uses, then assemble
    a :class:`VerificationResult` so the UI output shape is identical."""
    counter: Counter[str] = Counter(counts)
    top = counter.most_common(TOP_MEASUREMENTS_KEEP)
    top_measurements = tuple((bs, c) for bs, c in top)

    found_satisfying = False
    satisfying: dict[str, bool] | None = None
    for bits, _c in counter.most_common():
        candidate = prepared.encoder.bitstring_to_assignment(bits)
        if satisfies(prepared.cnf, candidate):
            found_satisfying = True
            satisfying = candidate
            break

    if mode == "consistency":
        contradiction = not found_satisfying
        counter_model: CounterModel | None = None
    else:  # entailment
        contradiction = found_satisfying
        counter_model = (
            CounterModel(assignment=satisfying)
            if found_satisfying and satisfying is not None
            else None
        )

    return VerificationResult(
        contradiction_found=contradiction,
        counter_model=counter_model,
        n_variables=prepared.n_qubits,
        n_clauses=prepared.n_clauses,
        n_grover_iterations=prepared.n_iterations,
        backend_name=backend_name,
        shots=prepared.shots,
        top_measurements=top_measurements,
    )


def _final_payload(
    label: str,
    mode: VerifyMode,
    job_id: str,
    backend_name: str,
    result: VerificationResult,
    wall: float,
    ibm_status: str,
    seconds_elapsed: int,
) -> dict[str, Any]:
    """Shape the completed-job dict the UI renders."""
    return {
        "status": "completed",
        "ibm_job_id": job_id,
        "ibm_job_url": IBM_CONSOLE_BASE + job_id,
        "ibm_status": ibm_status,
        "seconds_elapsed": seconds_elapsed,
        "example": label,
        "mode": mode,
        "backend": backend_name,
        "contradiction_found": result.contradiction_found,
        "counter_model": _format_counter_model(result, mode),
        "n_variables": result.n_variables,
        "n_clauses": result.n_clauses,
        "n_grover_iterations": result.n_grover_iterations,
        "shots": result.shots,
        "wall_clock_seconds": wall,
    }


def verify_on_ibm(label: str, mode: str) -> Iterator[dict[str, Any]]:
    """Submit the example to IBM and yield progressive status updates.

    First yield: ``status="submitted"`` with the IBM job_id and dashboard
    URL. Subsequent yields: ``status="in_progress"`` with the IBM-reported
    status and elapsed seconds. Final yield: ``status="completed"`` with
    the full verification result, or ``status="timeout"`` if the live
    poll loop exceeded :data:`LIVE_POLL_TIMEOUT_SECONDS`, or
    ``status="error"`` if something went wrong.
    """
    if label not in EXAMPLES:
        yield {"status": "error", "error": f"unknown example: {label}"}
        return
    try:
        narrow_mode = _coerce_mode(mode)
    except ValueError as exc:
        yield {"status": "error", "error": str(exc)}
        return

    if not IBM_AVAILABLE:
        yield {
            "status": "error",
            "error": (
                "IBM_QUANTUM_TOKEN and/or IBM_QUANTUM_INSTANCE are not set "
                "in this Space's Secrets. The IBM hardware path is "
                "unavailable; use the simulator button."
            ),
        }
        return

    start = time.monotonic()
    try:
        job, job_id, backend_name, prepared = _prepare_and_submit(label)
    except Exception as exc:
        yield {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return

    yield {
        "status": "submitted",
        "ibm_job_id": job_id,
        "ibm_job_url": IBM_CONSOLE_BASE + job_id,
        "ibm_status": "QUEUED",
        "seconds_elapsed": int(time.monotonic() - start),
        "backend": backend_name,
        "example": label,
        "mode": narrow_mode,
        "message": (
            "Job submitted. Polling every "
            f"{POLL_INTERVAL_SECONDS} s for up to "
            f"{LIVE_POLL_TIMEOUT_SECONDS // 60} minutes. If your browser "
            "loses the live stream before the job finishes, copy the "
            "Job ID from this output and use the 'Recover a previous "
            "job' panel below."
        ),
    }

    while True:
        elapsed = int(time.monotonic() - start)
        if elapsed > LIVE_POLL_TIMEOUT_SECONDS:
            yield {
                "status": "timeout",
                "ibm_job_id": job_id,
                "ibm_job_url": IBM_CONSOLE_BASE + job_id,
                "seconds_elapsed": elapsed,
                "message": (
                    "Live wait exceeded "
                    f"{LIVE_POLL_TIMEOUT_SECONDS} s without the job "
                    "completing. The job is still running on IBM. Use "
                    "the 'Recover a previous job' panel below with this "
                    "Job ID once it finishes (typical wall-clock 2-20 "
                    "minutes for free-tier queue + run)."
                ),
            }
            return

        time.sleep(POLL_INTERVAL_SECONDS)

        try:
            ibm_status = str(job.status())
        except Exception as exc:
            yield {
                "status": "error",
                "ibm_job_id": job_id,
                "ibm_job_url": IBM_CONSOLE_BASE + job_id,
                "seconds_elapsed": int(time.monotonic() - start),
                "error": f"polling failed: {type(exc).__name__}: {exc}",
            }
            return

        if ibm_status not in IBM_TERMINAL_STATUSES:
            yield {
                "status": "in_progress",
                "ibm_job_id": job_id,
                "ibm_job_url": IBM_CONSOLE_BASE + job_id,
                "ibm_status": ibm_status,
                "seconds_elapsed": int(time.monotonic() - start),
            }
            continue

        if ibm_status != "DONE":
            yield {
                "status": "error",
                "ibm_job_id": job_id,
                "ibm_job_url": IBM_CONSOLE_BASE + job_id,
                "ibm_status": ibm_status,
                "seconds_elapsed": int(time.monotonic() - start),
                "error": f"IBM job ended with status {ibm_status}",
            }
            return

        try:
            result_obj = job.result()
            pub_result = result_obj[0]
            raw_counts = pub_result.data.meas.get_counts()
        except Exception as exc:
            yield {
                "status": "error",
                "ibm_job_id": job_id,
                "ibm_job_url": IBM_CONSOLE_BASE + job_id,
                "seconds_elapsed": int(time.monotonic() - start),
                "error": f"result fetch failed: {type(exc).__name__}: {exc}",
            }
            return

        counts = _decode_counts(raw_counts)
        verification = _build_verification_result(counts, prepared, backend_name, narrow_mode)
        wall = round(time.monotonic() - start, 1)
        yield _final_payload(
            label, narrow_mode, job_id, backend_name, verification, wall, ibm_status, int(wall)
        )
        return


# --------------------------------------------------------------------------
# Manual recovery path - look up a previous job by ID
# --------------------------------------------------------------------------


def check_job_status(job_id: str) -> dict[str, Any]:
    """One-shot status query for a job already submitted (live stream lost)."""
    job_id = (job_id or "").strip()
    if not job_id:
        return {"error": "Enter a Job ID first."}

    if not IBM_AVAILABLE:
        return {
            "error": (
                "IBM_QUANTUM_TOKEN and/or IBM_QUANTUM_INSTANCE are not set "
                "in this Space's Secrets. Cannot query IBM Cloud."
            )
        }

    try:
        client = _build_runtime_client()
        service = client.get_service()
        job = service.job(job_id)
    except Exception as exc:
        return {"error": f"failed to look up job {job_id!r}: {type(exc).__name__}: {exc}"}

    try:
        ibm_status = str(job.status())
    except Exception as exc:
        return {
            "ibm_job_id": job_id,
            "ibm_job_url": IBM_CONSOLE_BASE + job_id,
            "error": f"status query failed: {type(exc).__name__}: {exc}",
        }

    if ibm_status not in IBM_TERMINAL_STATUSES:
        return {
            "status": "in_progress",
            "ibm_job_id": job_id,
            "ibm_job_url": IBM_CONSOLE_BASE + job_id,
            "ibm_status": ibm_status,
            "message": (
                "Job is still running on IBM. Refresh this panel "
                "periodically; IBM dashboards update at "
                f"{IBM_CONSOLE_BASE}{job_id}."
            ),
        }

    if ibm_status != "DONE":
        return {
            "status": "error",
            "ibm_job_id": job_id,
            "ibm_job_url": IBM_CONSOLE_BASE + job_id,
            "ibm_status": ibm_status,
            "error": f"IBM job ended with status {ibm_status}",
        }

    # Without the original CNF we cannot run the classical post-check or
    # report contradiction_found. Surface the raw counts so the user has
    # something to inspect; the full result is recoverable only by
    # re-submitting on the live path.
    try:
        result_obj = job.result()
        pub_result = result_obj[0]
        raw_counts = pub_result.data.meas.get_counts()
    except Exception as exc:
        return {
            "status": "error",
            "ibm_job_id": job_id,
            "ibm_job_url": IBM_CONSOLE_BASE + job_id,
            "error": f"result fetch failed: {type(exc).__name__}: {exc}",
        }

    counts = _decode_counts(raw_counts)
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_MEASUREMENTS_KEEP]
    return {
        "status": "completed",
        "ibm_job_id": job_id,
        "ibm_job_url": IBM_CONSOLE_BASE + job_id,
        "ibm_status": ibm_status,
        "top_measurements": top,
        "message": (
            "Raw counts retrieved. The classical SAT post-check requires "
            "the original CNF, which this fallback panel does not have; "
            "re-run on the live path to get a full contradiction_found "
            "verdict."
        ),
    }


# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------


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
                btn_sim = gr.Button("Verify on PennyLane simulator", variant="primary")
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
    gr.Markdown("### 4. Recover a previous job (fallback)")
    gr.Markdown(
        "If the live progress stream above disconnects (HF Spaces "
        "free-tier WebSockets sometimes drop on long-running tasks), "
        "copy the Job ID from the JSON output and check status here."
    )
    with gr.Row():
        job_id_input = gr.Textbox(
            label="IBM Job ID",
            placeholder="czxxx...",
            info="From a previous submission's JSON output",
        )
        btn_check = gr.Button("Check status by Job ID")
    status_output = gr.JSON(label="Job status")
    btn_check.click(check_job_status, inputs=[job_id_input], outputs=status_output)

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
        "**IBM hardware runs may take 2-20 minutes** due to free-tier "
        "queue plus on-device execution and result fetch. The live "
        f"button polls automatically every {POLL_INTERVAL_SECONDS} s "
        f"for up to {LIVE_POLL_TIMEOUT_SECONDS // 60} minutes; if the "
        "connection drops during a long run, use the 'Recover a "
        "previous job' panel above with the Job ID from your initial "
        "submission. Hardware credit is shared across users on the "
        "Open Plan (~10 minutes/month total)."
    )


if __name__ == "__main__":
    demo.launch()
