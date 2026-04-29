# QVerify

[![tests](https://github.com/Quantum-Labor/qverify/actions/workflows/tests.yml/badge.svg)](https://github.com/Quantum-Labor/qverify/actions/workflows/tests.yml)
[![lint](https://github.com/Quantum-Labor/qverify/actions/workflows/lint.yml/badge.svg)](https://github.com/Quantum-Labor/qverify/actions/workflows/lint.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Built with Gemma 4](https://img.shields.io/badge/Built%20with-Gemma%204-4285F4)](https://ai.google.dev/gemma)

QVerify checks logical reasoning steps from large language models using Grover's
search on a quantum simulator and on real IBM quantum hardware. Each reasoning
step is translated into a propositional logic formula, grounded in a finite
universe of constants, then verified for consistency against the chain of prior
premises.

A run on real hardware (IBM Heron r2 processor, ibm_fez backend, 156 qubits)
is recorded and reproducible: see [Hardware run](#hardware-run).

![Architecture diagram](assets/architecture.svg)

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

## Architecture

QVerify has four components, each in its own module:

| Module | Responsibility |
| --- | --- |
| `qverify.translator` | Natural-language to CNF using Gemma 4 E4B with constrained generation. Outputs a `TranslationResult` containing both a `CNF` and a `Universe` of declared constants. |
| `qverify.verifier.grounding` | Expands first-order CNF over a finite universe to a propositional CNF the verifier can handle. |
| `qverify.verifier` | Runs Grover's search on PennyLane simulator or IBM Quantum hardware. Supports two modes: `consistency` (default) checks if a formula is satisfiable, `entailment` checks if a step follows from premises. |
| `qverify.controller` | Orchestrates the full pipeline. Translates the problem statement to seed the initial universe, then for each reasoning step runs translate -> ground -> verify -> commit-or-retry. |

For a deeper walk-through, see [docs/architecture.md](docs/architecture.md).

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
tests/             # 370+ unit tests, GPU and hardware smoke tests gated
space/             # HuggingFace Space (Gradio demo)
assets/            # architecture diagram, badges
scripts/           # one-off diagnostics (gitignored)
```

## Roadmap

- v0.2: free-form natural-language reasoning. Translator gains a
  `translate_text(text)` method that splits multi-sentence input into single
  statements, translates each, and merges entities.
- v0.2: existential quantifier (`∃x`) support via Skolemization.
- v0.3: ProofWriter and FOLIO benchmarks. Comparison of vanilla Gemma 4 vs.
  Gemma 4 + QVerify on logical reasoning accuracy.
- v0.3: smarter grounding (only ground variables that co-occur with their
  predicates' constants) for larger universes.
- v0.4: Cirq backend for Google quantum simulators alongside PennyLane and
  Qiskit.

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

Serghei Brinza (<!-- TODO: github handle -->)
