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
    """Best-effort visitor IP from the Gradio Request (HF reverse proxy)."""
    try:
        headers = getattr(request, "headers", {}) or {}
        xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        if xff:
            return str(xff).split(",")[0].strip()
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
        _job, job_id, backend_name, _prepared = _prepare_and_submit(label)
    except Exception as exc:
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


def _lookup_job(service: Any, job_id: str) -> Any:
    """Find an IBM job by ID, with retry + jobs-list fallback for fresh jobs.

    IBM's runtime API sometimes has a propagation delay: a job created seconds
    ago may not yet be findable via service.job(id) even though it appears in
    the dashboard. Strategy: try direct lookup up to 3 times with exponential
    backoff, then scan the 50 most recent jobs as a fallback.
    """
    delay = 2
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            return service.job(job_id)
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
            delay *= 2

    # Fallback: scan recent jobs list
    try:
        for j in service.jobs(limit=50):
            if str(j.job_id()) == job_id:
                return j
    except Exception:
        pass  # if list scan also fails, surface the original direct-lookup error

    raise last_exc  # type: ignore[misc]


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
        job = _lookup_job(service, job_id)
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
        v0.2.0
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
        "Verifier accuracy against the PySAT Glucose3 oracle on three "
        "logic-reasoning datasets (ProofWriter, RuleTaker, FOLIO). "
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
