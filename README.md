# QVerify

**A quantum-assisted verifier for LLM reasoning, deployed end to end in the browser and on real IBM quantum hardware.**

> Each step a language model emits is translated into propositional CNF, grounded over a finite universe, and then checked for satisfiability by Grover's algorithm running on either a CPU simulator or IBM's 156-qubit Heron r2 processor. The verifier surfaces logical contradictions that no output-layer filter can catch.

**Live demo:** [huggingface.co/spaces/Laborator/qverify](https://huggingface.co/spaces/Laborator/qverify)

[![tests](https://github.com/Quantum-Labor/qverify/actions/workflows/tests.yml/badge.svg)](https://github.com/Quantum-Labor/qverify/actions/workflows/tests.yml)
[![lint](https://github.com/Quantum-Labor/qverify/actions/workflows/lint.yml/badge.svg)](https://github.com/Quantum-Labor/qverify/actions/workflows/lint.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Built with Gemma 4](https://img.shields.io/badge/Built%20with-Gemma%204-4285F4)](https://ai.google.dev/gemma)
[![Demo](https://img.shields.io/badge/HF%20Space-Laborator%2Fqverify-yellow)](https://huggingface.co/spaces/Laborator/qverify)
[![Release](https://img.shields.io/badge/release-v1.0.1-purple)](https://github.com/Quantum-Labor/qverify/releases/tag/v1.0.1)

## v1.0 status sheet

| | |
| --- | --- |
| **Release** | v1.0.1 — first stable, 2026-05-02 |
| **Verifier accuracy** | **100 %** on qverify-mini-50 (PySAT-validated, hand-crafted) |
| **Test count** | 433 unit tests, CI green (lint + ruff format + mypy strict, all clean) |
| **Real hardware** | [`d7q961poagoc73fj6oag`](https://quantum.cloud.ibm.com/workloads?search=d7q961poagoc73fj6oag) — `ibm_fez` (Heron r2, 156 qubits), 1024 shots, transpiled depth 360, 2026-05-01 |
| **Public deployment** | Hugging Face Space at [Laborator/qverify](https://huggingface.co/spaces/Laborator/qverify) — per-IP rate limit + global daily cap + IBM quota guard |
| **Pipeline** | `Translator (Gemma 4 E4B + outlines)` → `Grounding` → `Grover oracle synthesis` → `PennyLane simulator` ∥ `IBM Heron r2` |
| **License** | Apache 2.0 — datasets attributed under their own licenses ([benchmarks/LICENSE-DATA.md](benchmarks/LICENSE-DATA.md)) |
| **Position** | Project 1 of 3 in the Quantum Co-Processor research program |

## Why this is real engineering

- **Verified on real quantum hardware**, not just on a simulator. Single-shot
  reproducible job [`d7q961poagoc73fj6oag`](https://quantum.cloud.ibm.com/workloads?search=d7q961poagoc73fj6oag)
  on `ibm_fez` (156 qubits, IBM Heron r2). Anyone with an IBM Quantum account
  can open the URL and see the same circuit, the same shots, and the same
  measurement histogram.
- **Verifier accuracy is reported, not assumed.** The `qverify-mini-50`
  benchmark hand-crafts 50 SAT/UNSAT formulas covering propositional
  contradictions, modus ponens chains, transitivity, resolution,
  pigeonhole-tiny, grounded first-order, and AND/OR mixes. **Every label was
  cross-checked against PySAT Glucose3 before commit**, so the gold verdict
  is *independent* of QVerify's own implementation. The verifier scores 100 %
  (50/50) against this oracle.
- **Honest scope.** The two public NL benchmarks the harness also supports
  (ProofWriter, RuleTaker) are explicitly listed as *out of scope for v1.0*
  because their grounded CNFs (19-83 distinct atoms) blow past the 16-qubit
  PennyLane statevector ceiling. Their reports are checked in with
  `accuracy: n/a` rather than fabricated; smarter grounding lands in v1.1.
- **433 unit tests, all green in CI.** Lint (`ruff check`), formatting
  (`ruff format --check`), strict typecheck (`mypy qverify`) all clean on
  every push. The IBM hardware path has its own opt-in smoke test.
- **Production-grade public deployment.** The HF Space is gated by a
  three-layer safety stack — per-IP rate limit (1 IBM run / 5 min),
  global daily cap (5 / UTC day), IBM monthly-quota floor — so the project's
  free-tier budget cannot be drained by a script. Logic is in
  [`space/safety.py`](space/safety.py); 10 standalone tests.
- **Apache 2.0**, public org [github.com/Quantum-Labor](https://github.com/Quantum-Labor),
  proper dataset attribution.
- **Project 1 of 3** in the Quantum Co-Processor research program. This repo
  ships verifier, translator, and controller; the larger reasoning loop and
  the second/third companion projects are the upstream targets.

```
                              ┌────────────────────┐
  natural-language reasoning  │  Gemma 4 E4B       │
  step                        │  + outlines        │   first-order CNF
  ─────────────────────────►  │  (translator)      │  ────────────────►
                              └────────────────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │  ground_cnf()      │   propositional CNF
                              │  (grounding)       │  ────────────────►
                              └────────────────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │  Grover's search   │
                              │  (verifier)        │
                              └────────────────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ PennyLane (sim)    │
                              │  · OR ·            │
                              │ IBM Heron r2 (HW)  │
                              └────────────────────┘
```

A run on real hardware (IBM Heron r2 processor, ibm_fez backend, 156 qubits)
is recorded and reproducible: see [Hardware run](#hardware-run).

![Architecture diagram](assets/architecture.svg)

## Why this exists

Large language models in 2026 can produce fluent, step-by-step reasoning,
but there is no standard mechanism to check whether each step is logically
consistent with the steps before it. A model might assert "Tom has fur"
before ever establishing that Tom is a cat. The error is syntactically
invisible, semantically material, and not caught by any output-layer filter.

QVerify adds a formal consistency check after each reasoning step. The check
is automated (no human reads each step) and grounded in satisfiability theory
(NP-complete in the general case). The quantum component, Grover's search,
provides a quadratic speedup over classical brute-force SAT for the sizes that
matter in practice when verifying single reasoning steps against a growing set
of premises.

The current scope is small and honest: short chains of first-order premises
with a handful of constants, on a simulator or on a single IBM device. The
architecture is designed so that the `verify()` interface does not change as
the backend scales from simulator to 156-qubit hardware to future larger
processors.

## What works in v0.1

- Translation: natural-language premises and conclusions to first-order CNF using
  Gemma 4 E4B with grammar-constrained generation (outlines library).
- Grounding: first-order CNF with universal quantifiers expanded over a finite
  universe of constants, producing propositional CNF.
- Verification: Grover's search on PennyLane simulator (default) or IBM Quantum
  hardware (opt-in). Modes: consistency-checking (default) and entailment-checking.
- Controller: orchestrates the full pipeline with a problem pre-pass to seed
  the universe, then verifies each numbered step from a structured input.
- Real hardware: a single-shot smoke test runs on IBM Heron r2 and records the
  job ID for reference.

## What does not work in v0.1

- Free-form natural-language reasoning produced by Gemma 4 in real-time
  thinking mode. The thinking-phase output sometimes contains multi-sentence
  steps that the single-statement translator cannot parse. Roadmap item v0.2.
- Existential quantifiers (`∃x`). Translator emits an empty CNF with a warning.
- Equality and functions (e.g. `f(x) = g(y)`). Predicates only.
- Universes with more than 10-20 constants. Grounding is naive Cartesian product;
  larger inputs need smarter grounding strategies.

## Quick start

Install dependencies (Python 3.11+):

```bash
git clone https://github.com/Quantum-Labor/qverify
cd qverify
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite (excluding GPU and slow hardware tests):

```bash
.venv/bin/pytest -m "not slow and not gpu"
```

Verify a structured reasoning chain (simulator only, no GPU required for this
example because we use a stub LLM):

```python
from qverify.controller import StreamChunk, StubGemmaBackend, reason_with_verification
from qverify.verifier.backends import PennyLaneBackend

scene = [
    StreamChunk(text="Let me work through this.", phase="thinking"),
    StreamChunk(text="\n", phase="thinking"),
    StreamChunk(text="1. All cats have fur.\n", phase="answer"),
    StreamChunk(text="2. Tom is a cat.\n", phase="answer"),
    StreamChunk(text="3. Therefore Tom has fur.\n", phase="answer"),
    StreamChunk(text="\nYes.", phase="answer"),
]
llm = StubGemmaBackend(scripts=[scene])

result = reason_with_verification(
    problem="Premises: All cats have fur. Tom is a cat. Question: does Tom have fur?",
    llm=llm,
    verifier_backend=PennyLaneBackend(),
)
print(f"Answer: {result.final_answer}")
print(f"Verifications: {result.total_verifications}")
print(f"Counter-examples found: {result.total_contradictions_found}")
```

For real Gemma 4 E4B and PennyLane simulator (requires GPU and HF auth):

```bash
.venv/bin/pytest tests/test_controller_pipeline_sanity.py -v -m "slow and gpu" -s
```

## Hardware run

A single end-to-end smoke test ran on IBM Quantum's Heron r2 processor on
2026-04-28. Reproducibility:

| Property | Value |
| --- | --- |
| Backend | ibm_fez (Heron r2, 156 qubits) |
| Job ID | [d7o7dsqk4prs73dt4s6g](https://quantum.cloud.ibm.com/workloads?search=d7o7dsqk4prs73dt4s6g) |
| Date (UTC) | 2026-04-28 |
| Shots | 1024 |
| Grover iterations | 2 |
| Quantum runtime | ~6 s |
| Wall-clock | ~54 s (queue + transpile + run) |

To reproduce on your own IBM Quantum account, populate `.env` with
`IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_INSTANCE`, then:

```bash
.venv/bin/pytest tests/test_verifier_ibm_smoke.py -v -m slow -s
```

## Performance characteristics

These are observed numbers from the v0.1 development cycle, not benchmarks.
They are provided so the reader can calibrate expectations, not as claims.

| Scenario | Backend | Wall-clock | Notes |
| --- | --- | --- | --- |
| 2-variable CNF, 1 Grover iteration | PennyLane simulator | < 10 ms | Statevector simulation |
| 6-variable CNF, 6 Grover iterations | PennyLane simulator | ~50 ms | Typical step in a 4-premise chain |
| 10-variable CNF, 8 Grover iterations | PennyLane simulator | ~300 ms | Near practical limit for the simulator |
| 3-variable CNF, 2 Grover iterations | IBM Heron r2 (ibm_fez) | ~6 s quantum runtime, ~54 s wall-clock | Queue + transpile + run + fetch |

The simulator path scales poorly past ~12 variables because statevector
simulation requires 2^n complex amplitudes. At 12 variables that is 4,096
amplitudes; at 20 it is 1,048,576. For the current use case (verifying
individual reasoning steps against 3-8 premises over a handful of constants),
the grounded CNF typically has 4-10 variables, which the simulator handles in
milliseconds.

The IBM hardware path is bottlenecked almost entirely by queue time on the
free tier. The circuit itself ran in approximately 6 seconds on Heron r2; the
remaining 48 seconds were queue, transpilation, and result fetch. Free-tier
accounts share roughly 10 minutes of quantum time per month across all users
on the Open Plan.

Classical SAT solvers (e.g. MiniSAT) solve these instances in microseconds.
The quantum path is slower on today's hardware and at these problem sizes.
The `verify()` interface is designed so that as hardware scales, the client
code does not change.

## Benchmarks

The harness in [qverify/eval/](qverify/eval/) runs the verifier against
ProofWriter and RuleTaker and compares each output against the PySAT
Glucose3 oracle. See [docs/benchmarks.md](docs/benchmarks.md) for the full
methodology.

Currently benchmarked on ProofWriter and RuleTaker. See
[benchmarks/LICENSE-DATA.md](benchmarks/LICENSE-DATA.md) for dataset
attribution and the FOLIO exclusion rationale.

**Verified end-to-end on the qverify-mini-50 hand-crafted suite**
(50 examples covering propositional contradictions, modus ponens chains,
transitivity, pigeonhole-tiny, AND/OR mixes, and grounded first-order
formulas — every label cross-checked against PySAT before commit):

| Metric | Value |
| --- | --- |
| Examples | 50 (25 SAT / 25 UNSAT) |
| Accuracy vs PySAT oracle | **100 %** (50 / 50) |
| Avg verify time (simulator) | 3.3 s |
| P95 verify time | 0.32 s |
| Atoms per example | 1 – 9 (under the 16-qubit cap) |
| Skipped | 0 |

The full per-example report and charts live at
[benchmarks/results/qverify_mini_simulator/](benchmarks/results/qverify_mini_simulator/).
The dataset itself is at
[benchmarks/qverify_mini/dataset.json](benchmarks/qverify_mini/dataset.json),
with a category-by-category breakdown in
[benchmarks/qverify_mini/README.md](benchmarks/qverify_mini/README.md).

### Benchmark scope

QVerify v1.0 ships **one** benchmark with reportable accuracy:
`qverify-mini-50`. The two natural-language datasets that the harness
also supports are listed here for completeness with their honest scope:

| Dataset | Status in v1.0 | Why |
| --- | --- | --- |
| qverify-mini-50 | **In scope.** 100 % accuracy on 50 hand-crafted examples. | CNFs sized to the simulator (1-9 atoms); labels validated by PySAT. |
| ProofWriter (depth-1) | Out of scope. | Grounded CNFs reach 19-83 atoms — beyond the 16-qubit PennyLane ceiling (`MAX_VARIABLES = 16`). The translation path runs (92 / 100 examples translate cleanly) but every successful translation hits `n_skipped_too_large`. |
| RuleTaker (depth-1) | Out of scope. | Same atom-count blow-up after grounding (25-54 atoms). Tracked for v1.1 once smarter grounding (Roadmap v0.3) lands. |

The intention is honest: hand-crafted SAT benchmarks demonstrate verifier
correctness *now*; the public NL datasets exercise translation but starve
the verifier until grounding gets smarter. Both axes have to scale before
we can report a meaningful NL-benchmark accuracy.

The full ProofWriter / RuleTaker reports (with `accuracy: n/a`) are
checked into [benchmarks/results/](benchmarks/results/) for transparency.

To reproduce the runs above, first download the datasets:

```bash
python scripts/download_datasets.py --datasets proofwriter,ruletaker
```

Then run the benchmarks (requires a CUDA GPU for `--translate gemma-e2b`;
omit the flag and use a fixture path for CPU-only verification):

```bash
python scripts/run_benchmarks.py --dataset proofwriter --backend simulator \
    --max-examples 100 --translate gemma-e2b \
    --output benchmarks/results/proofwriter_simulator
python scripts/run_benchmarks.py --dataset ruletaker --backend simulator \
    --max-examples 200 --translate gemma-e2b \
    --output benchmarks/results/ruletaker_simulator
```

Charts (qubit-count distribution, latency, accuracy) and `report.json`
land under each `--output` directory.

## Architecture

QVerify has four components, each in its own module:

| Module | Responsibility |
| --- | --- |
| `qverify.translator` | Natural-language to CNF using Gemma 4 E4B with constrained generation. Outputs a `TranslationResult` containing both a `CNF` and a `Universe` of declared constants. |
| `qverify.verifier.grounding` | Expands first-order CNF over a finite universe to a propositional CNF the verifier can handle. |
| `qverify.verifier` | Runs Grover's search on PennyLane simulator or IBM Quantum hardware. Supports two modes: `consistency` (default) checks if a formula is satisfiable, `entailment` checks if a step follows from premises. |
| `qverify.controller` | Orchestrates the full pipeline. Translates the problem statement to seed the initial universe, then for each reasoning step runs translate -> ground -> verify -> commit-or-retry. |

For a deeper walk-through, see [docs/architecture.md](docs/architecture.md).

## Design decisions

**Consistency checking instead of entailment.** The verifier runs in
consistency mode by default: it checks that `premises ∧ step` is satisfiable.
A satisfying assignment means the step is consistent with everything established
so far; UNSAT means the step contradicts a premise. This is cheaper to compute
than full entailment (`premises ∧ ¬step` UNSAT) and catches the failure mode
that matters most in practice: a step that contradicts what the model already
said.

**First-order input, propositional execution.** The translator and grounder
together handle the first-order layer; Grover's circuit sees only propositional
CNF. This keeps the quantum component simple and means all circuit logic (qubit
count, oracle construction, iteration count) is determined at grounding time
from the variable count, not from the predicate structure.

**Grammar-constrained generation for translation.** The translator uses
outlines to force Gemma 4 E4B to emit well-formed first-order logic JSON on
every call. Without constrained generation, 4B-parameter models frequently
emit malformed outputs or substitute constants where free variables are needed.
With it, the translation step is reliable enough to be unattended inside the
controller loop.

**Stateless verify() interface.** The `verify()` function takes a CNF and
returns a `VerificationResult`. It has no internal state, no threading, and
no side effects beyond calling the backend. This makes it straightforward to
swap backends (PennyLane simulator for development, IBM hardware for validation)
and to test each layer independently.

**Retry on contradiction.** When the verifier finds that a step contradicts
established premises, the controller formats the counter-model (or a generic
"contradicts established premises" message in consistency mode) as a prompt
and asks the model to rewrite the step. The retry count is bounded. If the
model cannot produce a consistent step, the pipeline reports the failure rather
than silently committing an inconsistent step.

## Why Gemma 4

The translator uses Gemma 4 E4B specifically for two reasons.

First, the thinking-mode output structure (channel-tagged reasoning followed by
the final answer) gives a clean separation between intermediate steps and
conclusions, which the controller relies on for step-by-step verification.

Second, behaviour with grammar-constrained generation (via the outlines library)
is reliable on first-order logic schemas at the 4B parameter scale. The model
correctly emits free variables for universal statements (`Cat(x)` rather than
`Cat(Cat)`), where smaller models often substitute constants. Tested on
`google/gemma-4-E4B-it` from Hugging Face.

## Why a quantum verifier

For NP-complete satisfiability checks, Grover's algorithm offers a quadratic
speedup over classical brute force in the worst case. On current hardware and
small CNF instances, classical SAT solvers remain faster in wall-clock terms.
The value of QVerify today is not throughput but the architecture: the same
`verify()` interface runs on a simulator during development and on real
hardware for validation. As quantum hardware scales, larger verification
queries become feasible without changing client code.

The hardware run on Heron r2 is documented above as a reproducibility check,
not a performance benchmark.

## Repository layout

```
qverify/
  controller/      # orchestration, step extraction, feedback loop
  translator/      # NL to CNF with constrained generation
  verifier/        # Grover's search, grounding, encoding
  utils/           # configuration, logging, model IDs
docs/              # architecture, grounding, controller, hardware notes
tests/             # 433 unit tests, GPU and hardware smoke tests gated
space/             # HuggingFace Space (Gradio demo)
assets/            # architecture diagram, badges
scripts/           # one-off diagnostics (gitignored)
```

## Roadmap

**v0.2 (next release)**

- Free-form natural-language reasoning. Translator gains a `translate_text(text)`
  method that splits multi-sentence input into single statements, translates each,
  and merges entities. This unblocks the thinking-mode failure case documented in
  "What does not work in v0.1".
- Existential quantifier (`∃x`) support via Skolemization. The translator currently
  emits an empty CNF with a warning for existential statements; Skolemization
  replaces them with fresh Skolem constants before grounding.

**v0.3**

- ProofWriter and FOLIO benchmarks. Structured evaluation of vanilla Gemma 4 vs.
  Gemma 4 + QVerify on logical reasoning accuracy, to measure whether the
  verification loop materially reduces step-level errors on standard datasets.
- Smarter grounding. The current grounder expands all free variables over all
  constants (Cartesian product). For predicates with arity 2 and 10 constants
  this produces 100 ground atoms. A predicate-aware grounder would restrict
  each variable to the constants that appear with it in the premises, reducing
  the clause count substantially for large universes.

**v0.4**

- Cirq backend for Google quantum simulators alongside PennyLane and Qiskit.
  The `verify()` interface already abstracts the backend; adding Cirq is
  primarily a circuit translation task.

## Acknowledgments

Built on Gemma 4 by Google DeepMind, Qiskit by IBM Quantum, PennyLane by Xanadu,
and outlines by .txt. Hardware verification on IBM Quantum's Heron r2 processor.

## Citation

If you reference QVerify in academic work:

```bibtex
@misc{brinza2026qverify,
  author       = {Serghei Brinza},
  title        = {QVerify: quantum-assisted verification of LLM reasoning},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Quantum-Labor/qverify}},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Author


Serghei Brinza ([@SergheiBrinza](https://github.com/SergheiBrinza))
