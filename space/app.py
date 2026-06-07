"""QVerify HuggingFace Space - verifier-only demo, runs on CPU Basic.

This Space exposes the QVerify *verifier* component: a CNF is fed
straight to Grover's search, on either the PennyLane simulator (local
to this Space's CPU container) or IBM Heron r2 hardware (delegated to
IBM Cloud over the network, account credentials read from Space
Secrets).

The simulator path is synchronous and returns in milliseconds. The
IBM hardware path takes 2-20 minutes wall-clock (queue + transpile +
execution + result fetch), well past Gradio's WebSocket timeout on
HF Spaces' free tier. The IBM handler therefore submits the job and
returns immediately with the IBM job_id; the run's status and final
measurement histogram are tracked on the IBM Quantum Workloads
dashboard (linked in the result). Live in-Space polling and a
recover-by-job-id panel are planned but are not part of this
verifier-only build.

The translator (Gemma 4 E4B with grammar-constrained generation)
is intentionally not loaded here: it needs a GPU, which CPU Basic
does not provide. The full pipeline runs locally from the GitHub
repository, see https://github.com/Quantum-Labor/qverify.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, cast
from typing import Literal as _Literal

import gradio as gr

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.utils.ibm_client import IBMRuntimeClient
from qverify.verifier import verify
from qverify.verifier._universe import Universe
from qverify.verifier.backends import PennyLaneBackend
from qverify.verifier.encoding import AtomEncoder
from qverify.verifier.grounding import ground_cnf
from qverify.verifier.grover import MAX_VARIABLES, optimal_iterations
from qverify.verifier.grover_circuit import build_grover_qiskit_circuit
from qverify.verifier.types import VerificationResult

VerifyMode = _Literal["entailment", "consistency"]

_QV_THEME = gr.themes.Base(
    primary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0F0F1A",
    body_text_color="#E5E7EB",
    body_text_color_subdued="#A1A1AA",
    background_fill_primary="#1A1A2E",
    background_fill_secondary="#0F0F1A",
    border_color_primary="#2A2440",
    button_primary_background_fill="#5B21B6",
    button_primary_background_fill_hover="#7C3AED",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#1F1B33",
    button_secondary_background_fill_hover="#2A2440",
    button_secondary_text_color="#E5E7EB",
    block_background_fill="#15152A",
    block_border_color="#2A2440",
    input_background_fill="#15152A",
    input_border_color="#2A2440",
)

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


# -- Safety: per-IP rate limit + global daily cap + IBM quota guard ---------


# Loaded by file path because space/ is not a Python package on the
# Hugging Face Space deployment (app.py is at the Space root).
def _load_safety_module() -> Any:
    import importlib.util as _ilu
    import sys as _sys

    if "_qv_safety" in _sys.modules:
        return _sys.modules["_qv_safety"]

    here = Path(__file__).resolve().parent
    for candidate in (here / "safety.py", here / "space" / "safety.py"):
        if candidate.exists():
            spec = _ilu.spec_from_file_location("_qv_safety", candidate)
            assert spec is not None and spec.loader is not None
            module = _ilu.module_from_spec(spec)
            # Register before exec so @dataclass can resolve module-level
            # forward references via sys.modules lookup.
            _sys.modules["_qv_safety"] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("safety.py not found next to app.py")


_safety_mod = _load_safety_module()
_RATE_LIMITER = _safety_mod.RateLimiter(
    window_seconds=300,
    daily_cap=5,
    quota_floor_seconds=60,
    persist_path=_safety_mod.default_persist_path(),
)


def _client_ip(request: Any) -> str:
    """Best-effort visitor IP from the Gradio Request (HF reverse proxy).

    Reads the *last* X-Forwarded-For hop rather than the first. A client can
    forge the leftmost entries by sending its own X-Forwarded-For header, but
    the rightmost entry is appended by the trusted proxy in front of the Space,
    so it is the hardest for a caller to spoof. Taking the first hop let an
    attacker present a fresh IP per request and bypass the per-IP throttle.
    """
    try:
        headers = getattr(request, "headers", {}) or {}
        xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        if xff:
            return str(xff).split(",")[-1].strip()
        return str(getattr(request, "client", None) and request.client.host) or "unknown"
    except Exception:
        return "unknown"


# IBM monthly remaining-seconds probe. The API call is best-effort: on the
# Open Plan service.usage() may not return a usable value, so we degrade
# to "unknown" (None) which RateLimiter treats as no quota gating.
_quota_cache: dict[str, Any] = {"remaining_seconds": None, "checked_at": 0.0}


def _quota_remaining_seconds() -> float | None:
    """Return cached IBM remaining-seconds, refreshing at most every hour."""
    if not IBM_AVAILABLE:
        return None
    now = time.monotonic()
    if now - _quota_cache["checked_at"] < 3600 and _quota_cache["remaining_seconds"] is not None:
        return float(_quota_cache["remaining_seconds"])
    try:
        client = _build_runtime_client()
        usage = client.get_service().usage()
    except Exception:
        return _quota_cache["remaining_seconds"]
    remaining: float | None = None
    if isinstance(usage, dict):
        for key in ("remaining_seconds", "remaining", "balance"):
            if key in usage:
                try:
                    remaining = float(usage[key])
                    break
                except (TypeError, ValueError):
                    continue
    _quota_cache["checked_at"] = now
    _quota_cache["remaining_seconds"] = remaining
    return remaining


def _safety_status_md() -> str:
    """Live counter shown next to the IBM button."""
    from datetime import datetime as _dt

    if not IBM_AVAILABLE:
        return ""
    remaining = _RATE_LIMITER.daily_remaining(now=_dt.now(UTC))
    cap = _RATE_LIMITER.daily_cap()
    return (
        f"**{remaining} / {cap}** IBM runs remaining today · resets at "
        "midnight UTC. Per-visitor rate limit: 1 IBM run per 5 minutes."
    )


def _load_qverify_mini_summary() -> str:
    """Render the qverify-mini-50 hero card from the in-repo report.json.

    Returns a static Markdown card with accuracy, avg verify time, and
    example count so visitors see real numbers right under the intro
    copy. If the report file is missing (e.g. someone running locally
    before the first run) we fall back to a one-line placeholder.
    """
    import json as _json

    here = Path(__file__).resolve().parent
    candidates = (
        here / "benchmarks" / "results" / "qverify_mini_simulator" / "report.json",
        here.parent / "benchmarks" / "results" / "qverify_mini_simulator" / "report.json",
    )
    for p in candidates:
        if p.exists():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            acc = float(data.get("accuracy", 0.0)) * 100
            avg = float(data.get("avg_seconds", 0.0))
            n = int(data.get("n_examples", 0))
            return (
                "## qverify-mini-50 — verifier accuracy on the simulator\n\n"
                "Hand-crafted SAT/UNSAT benchmark, every label cross-checked "
                "against the PySAT Glucose3 oracle.\n\n"
                "| Metric | Value |\n"
                "| --- | --- |\n"
                f"| Accuracy | **{acc:.1f}%** |\n"
                f"| Avg verify time | {avg:.2f}s |\n"
                f"| Examples | {n} (25 SAT / 25 UNSAT) |\n"
                "\n"
                "Source: `benchmarks/qverify_mini/dataset.json` · "
                "[Methodology](https://github.com/Quantum-Labor/qverify/blob/main/benchmarks/qverify_mini/README.md)"
            )
    return "_qverify-mini-50 report not yet generated._"


def _load_benchmark_summaries() -> str:
    """Read every benchmarks/results/*/report.json checked into the repo and
    render a summary Markdown table. Called once at Space load time."""
    import json
    from pathlib import Path

    # On the HF Space app.py sits at the repo root, so benchmarks/ is one
    # directory down from app.py. In the local repo app.py lives under
    # space/, so benchmarks/ is two directories up. Try both layouts.
    here = Path(__file__).resolve().parent
    for candidate in (here / "benchmarks" / "results", here.parent / "benchmarks" / "results"):
        if candidate.exists():
            results_root = candidate
            break
    else:
        return "_No benchmark reports checked in yet._"

    rows: list[str] = []
    for report_path in sorted(results_root.glob("*/report.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            f"| {data.get('dataset', '?')} "
            f"| {data.get('backend', '?')} "
            f"| {data.get('n_examples', 0)} "
            f"| {data.get('accuracy', 0.0) * 100:.1f}% "
            f"| {data.get('avg_seconds', 0.0):.3f} s "
            f"| {data.get('p95_seconds', 0.0):.3f} s |"
        )

    if not rows:
        return "_No benchmark reports checked in yet._"

    header = (
        "| Dataset | Backend | Examples | Accuracy | Avg | P95 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    return header + "\n".join(rows)


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


def _prepare_and_submit(label: str, shots: int = 1024) -> tuple[str, str]:
    """Ground, encode, transpile, and submit the example to IBM hardware.

    Returns ``(job_id, backend_name)``. The blocking ``job.result()`` is
    deliberately not awaited here: the handler returns the job_id
    immediately and the run is tracked on the IBM Workloads dashboard.
    """
    ex = EXAMPLES[label]
    grounded = ground_cnf(ex.cnf, ex.universe)

    encoder = AtomEncoder(grounded)
    n_qubits = encoder.n_qubits
    if n_qubits > MAX_VARIABLES:
        raise ValueError(
            f"CNF has {n_qubits} variables; the Space accepts at most "
            f"{MAX_VARIABLES}. Use a smaller example."
        )
    encoded_clauses = encoder.encode_clauses()
    n_iterations = optimal_iterations(n_qubits)

    circuit = build_grover_qiskit_circuit(encoded_clauses, n_qubits, n_iterations)

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
    return str(job.job_id()), str(backend_name)


def verify_on_ibm(label: str, mode: str, request: gr.Request | None = None) -> dict[str, Any]:
    """Submit job to IBM, return Job ID immediately (no polling).

    Gated by :class:`safety.RateLimiter`: per-IP throttle (1 per 5 min),
    global daily cap (5 per UTC day), and IBM monthly-quota floor (60 s).
    """
    from datetime import datetime as _dt

    if label not in EXAMPLES:
        return {"status": "error", "error": f"unknown example: {label}"}

    try:
        narrow_mode = _coerce_mode(mode)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    if not IBM_AVAILABLE:
        return {
            "status": "error",
            "error": "IBM credentials not configured in Space Secrets.",
        }

    ip = _client_ip(request) if request is not None else "local-dev"
    verdict = _RATE_LIMITER.check_and_register(
        ip=ip,
        now=_dt.now(UTC),
        quota_remaining_seconds=_quota_remaining_seconds(),
    )
    if not verdict.allowed:
        return {
            "status": "blocked",
            "reason": verdict.reason,
            "message": verdict.detail,
            "daily_remaining": verdict.daily_remaining,
            "daily_cap": verdict.daily_cap,
        }

    start = time.monotonic()
    try:
        job_id, backend_name = _prepare_and_submit(label)
    except Exception as exc:
        # The rate-limit slot was committed on allow; the submit failed, so
        # give it back rather than charging the visitor for a no-op.
        _RATE_LIMITER.refund(ip=ip, now=_dt.now(UTC))
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    elapsed = int(time.monotonic() - start)
    return {
        "status": "submitted",
        "ibm_job_id": job_id,
        "ibm_job_url": IBM_CONSOLE_BASE + job_id,
        "backend": backend_name,
        "example": label,
        "mode": narrow_mode,
        "seconds_elapsed": elapsed,
        "daily_remaining": verdict.daily_remaining,
        "daily_cap": verdict.daily_cap,
        "message": (
            f"Job {job_id} submitted to {backend_name}. "
            f"{verdict.daily_remaining}/{verdict.daily_cap} IBM runs left today. "
            "View status at https://quantum.cloud.ibm.com/workloads"
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


def _load_logo_svg() -> str:
    """Inline the Quantum Labor logo SVG from the repo so the Space hero
    section renders without an external asset request."""
    here = Path(__file__).resolve().parent
    for candidate in (
        here / "assets" / "quantum_labor_logo.svg",
        here.parent / "assets" / "quantum_labor_logo.svg",
    ):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""  # graceful fallback: no logo, just text header


_LOGO_SVG = _load_logo_svg()

_HERO_HTML = f"""
<div style="display:flex;align-items:center;gap:24px;padding:24px 0 8px 0;">
  <div style="flex:0 0 auto;width:120px;height:120px;">{_LOGO_SVG}</div>
  <div>
    <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
      <h1 style="margin:0;font-size:32px;font-weight:800;letter-spacing:-0.02em;color:#E5E7EB;">
        QVerify
      </h1>
      <span style="font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;
                   color:#06B6D4;border:1px solid #06B6D4;padding:2px 8px;border-radius:999px;">
        v1.0.1
      </span>
      <span style="font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;
                   color:#10B981;border:1px solid #10B981;padding:2px 8px;border-radius:999px;">
        stable · 433 tests · CI green
      </span>
      <span style="font-family:JetBrains Mono,monospace;font-size:12px;color:#A78BFA;">
        Quantum Labor · Project 1 of 3
      </span>
    </div>
    <div style="margin-top:6px;color:#A1A1AA;font-size:18px;font-weight:500;">
      A Quantum Logic Verifier for AI Reasoning
    </div>
  </div>
</div>
"""

_INTRO_MD = """
## What is this?

Modern language models reason in plain English — but plain English can hide
contradictions. **QVerify catches them using a real quantum computer.**

This Space lets you watch **Grover's algorithm** — the most famous quantum
search algorithm in existence — hunt for logical contradictions in real time.

## How it works (in 30 seconds)

1. **PICK** an example reasoning chain (cat-fur logic, etc.)
2. **CHOOSE** a verification mode (consistency or entailment)
3. **RUN** it on either:
   - The **PennyLane SIMULATOR** — instant, runs on this server's CPU
   - **IBM HERON r2** — a real superconducting quantum processor in New York

## What you'll see

- The **CNF formula** being checked (a logical sentence in conjunctive normal form)
- The **Grover circuit** (live qubit count, gate count, Grover iterations)
- The **result**: contradiction found / not found, plus a counter-model if found
- For IBM runs: a **real IBM job ID** you can verify on
  [quantum.ibm.com](https://quantum.cloud.ibm.com/workloads)

## Why this matters

This is the verifier component of a three-project research program
(**Quantum Co-Processor**) exploring how quantum computers can serve as
specialized cognitive co-processors for large language models. QVerify
shows that Grover's algorithm — invented in 1996 — is finally usable on
real hardware for a non-trivial AI safety task in 2026.

The full pipeline (translator + verifier + reasoner) runs from the
[GitHub repository](https://github.com/Quantum-Labor/qverify) and uses
Google's **Gemma 4** family for the natural-language to CNF translation
step. This Space exposes the verifier alone, since the translator needs
a GPU.
"""

_RESULT_EXPLAINER_MD = """
The JSON above is the verifier's full output. Reading it:

- **`status`** — `submitted` / `in_progress` / `completed` for IBM runs;
  `completed` for simulator runs (synchronous).
- **`backend`** — `default.qubit` for simulator, `ibm_fez` / `ibm_kingston`
  for IBM Heron r2.
- **`mode`** — `consistency` checks if the formula is satisfiable;
  `entailment` checks if the negation is satisfiable (yielding a
  counter-model when the conclusion does NOT follow).
- **`contradiction_found`** — the verdict. `false` means the formula is
  consistent (in consistency mode) or entailed (in entailment mode).
- **`n_variables`** — Grover qubit count, which equals the number of
  distinct propositional atoms after grounding.
- **`n_grover_iterations`** — `round(π/4 · √(N/M))`, the optimal number
  of amplitude-amplification iterations for `N = 2^n` states and `M`
  satisfying assignments (`M = 1` by default).
- **`top_measurements`** — the most-frequent bitstrings observed across
  `shots` measurements. The classical post-check walks these in order
  until one classically satisfies the CNF.
- **For IBM runs**: `ibm_job_id` and `ibm_job_url` link to the live job
  record. Anyone with an IBM Quantum account can open the URL.
"""


with gr.Blocks(title="QVerify · Quantum Logic Verifier") as demo:
    gr.HTML(_HERO_HTML)
    gr.Markdown(_INTRO_MD)
    gr.Markdown("---")
    gr.Markdown(_load_qverify_mini_summary())
    gr.Markdown("---")
    gr.Markdown("# Try it now")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Pick a reasoning example")
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

            gr.Markdown("### 2. What should the verifier check?")
            mode_radio = gr.Radio(
                label="Verification mode",
                choices=["consistency", "entailment"],
                value="consistency",
                info=(
                    "consistency — Is this set of facts internally consistent? "
                    "(No contradiction.) "
                    "entailment — Do these premises logically force the conclusion?"
                ),
            )

            gr.Markdown("### 3. Run")
            with gr.Row():
                btn_sim = gr.Button(
                    "RUN ON SIMULATOR (free, instant)",
                    variant="primary",
                )
                btn_hw = gr.Button(
                    "RUN ON QUANTUM COMPUTER (IBM Heron r2, ~2 min)",
                    variant="secondary",
                    interactive=IBM_AVAILABLE,
                )
            gr.Markdown(_ibm_status_md)
            safety_md = gr.Markdown(_safety_status_md())
            gr.Markdown(
                "Submission returns a Job ID immediately. The circuit then "
                "runs on IBM hardware (typically 1-3 minutes for queue + "
                "execution). Track live status and retrieve results at "
                "[IBM Quantum Workloads](https://quantum.cloud.ibm.com/workloads)."
            )

        with gr.Column(scale=1):
            gr.Markdown("### Result")
            output = gr.JSON(label="Verification result")
            with gr.Accordion(
                "What just happened? (read this if the JSON looks scary)", open=False
            ):
                gr.Markdown(_RESULT_EXPLAINER_MD)

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
    ).then(
        lambda: _safety_status_md(),
        inputs=None,
        outputs=safety_md,
    )

    gr.Markdown("---")
    gr.Markdown(
        "### IBM Hardware Jobs\n\n"
        "Job execution happens on IBM Quantum hardware. After clicking "
        "**RUN ON QUANTUM COMPUTER**, copy the Job ID from the Result and "
        "view live status at the "
        "[IBM Quantum Workloads dashboard](https://quantum.cloud.ibm.com/workloads). "
        "All jobs are publicly verifiable there."
    )
    gr.Markdown("---")
    gr.Markdown(
        "### Benchmarks\n\n"
        "Verifier accuracy against the PySAT Glucose3 oracle. The "
        "headline suite is the hand-crafted qverify-mini-50; ProofWriter and "
        "RuleTaker are also wired up but out of scope for v1.0 (their grounded "
        "CNFs exceed the simulator qubit ceiling). FOLIO is excluded for "
        "licensing reasons. "
        "Reports are generated by `scripts/run_benchmarks.py` and "
        "checked into `benchmarks/results/`. See "
        "[docs/benchmarks.md]"
        "(https://github.com/Quantum-Labor/qverify/blob/main/docs/benchmarks.md)"
        " for methodology.\n\n" + _load_benchmark_summaries()
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
        "**IBM hardware runs are submitted via your IBM Quantum "
        "free-tier account** (Open Plan, approximately 10 minutes of "
        "quantum time per month, shared across users). The Space "
        "orchestrates the call; the actual circuit executes on a "
        "least-busy IBM Heron r2 backend (156 qubits, e.g. `ibm_kingston` "
        "or `ibm_fez`) via IBM Cloud. The submission returns immediately "
        "with a Job ID; results are retrieved at the IBM Quantum Workloads "
        "dashboard. Previously verified hardware runs are documented "
        "[here](https://github.com/Quantum-Labor/qverify#hardware-run)."
    )


if __name__ == "__main__":
    demo.launch(theme=_QV_THEME)
