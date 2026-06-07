---
title: QVerify
emoji: 🔬
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
hardware: cpu-basic
tags:
  - quantum-computing
  - qiskit
  - pennylane
  - llm-verification
  - logic
  - grover
  - auto-deploy
---

# QVerify — verifier-only Space

## What this demonstrates

This Space runs the QVerify verifier component: a propositional CNF is
grounded over a declared universe of constants, encoded into a Grover
circuit, and executed on either a CPU-side PennyLane statevector simulator
or IBM Quantum's Heron r2 processor (156 qubits, `ibm_fez` backend).

Three pre-built CNF examples are provided via a dropdown — a two-atom
propositional formula, a single ground first-order atom, and a universally
quantified rule grounded over two constants. Selecting an example shows the
formula in clause-normal form, the universe, and a plain-language description
of what the verifier should return. Two verification modes are available:

- **consistency**: checks that `premises ∧ step` is satisfiable. A satisfying
  assignment means the formula is consistent; UNSAT means the clauses
  contradict each other.
- **entailment**: checks that `premises ∧ ¬step` is satisfiable. A satisfying
  assignment is a counter-model showing the step is not entailed.

The classical post-check walks the most-frequent measured bitstrings until
one satisfies the CNF. This mirrors the post-check used in the full local
pipeline.

IBM hardware runs take 2-20 minutes wall-clock on the free tier (queue +
transpile + execution + result fetch). The Space submits the job and polls
every 8 seconds, yielding the job ID immediately so you can copy it before
the browser session ends. A fallback "recover by job ID" panel lets you
retrieve the result from any previous submission even if the live stream
dropped mid-run.

IBM hardware credit is shared across all users on the Open Plan
(approximately 10 minutes of quantum time per month total). The simulator
path runs locally in the Space container and has no external cost.

A **Verified on IBM Quantum Hardware** gallery further down the page lists past
Grover verification runs executed on Heron r2 devices (`ibm_fez`,
`ibm_kingston`), each with its Job ID linked to the public IBM Quantum
Workloads dashboard so anyone can confirm it. Two runs are featured with their
full CNF, mode, verdict, and measured counts; the remainder are listed as
verified runs with backend, date, shots, and circuit depth.

## What is not in this Space

The translator — Gemma 4 E4B with grammar-constrained generation — converts
natural-language premises and conclusions into first-order CNF. It requires
a GPU and is not loaded here; CPU Basic Spaces cannot host a 4-billion-
parameter model. The full pipeline (translator + grounding + verifier) runs
locally from the GitHub repository and requires a CUDA-capable GPU and a
Hugging Face account with access to `google/gemma-4-E4B-it`.

The grounding step (Cartesian-product expansion of first-order CNF over a
finite universe) is included here and runs automatically before the Grover
circuit is constructed.

## Resources

- [GitHub repository](https://github.com/Quantum-Labor/qverify) — source,
  local install instructions, test suite, architecture documentation
- [Hardware run](https://github.com/Quantum-Labor/qverify#hardware-run) —
  reproducibility record from the 2026-04-28 run on IBM Heron r2 (job ID
  `d7o7dsqk4prs73dt4s6g`, 1024 shots, 2 Grover iterations, ~6 s quantum
  runtime)
- [docs/architecture.md](https://github.com/Quantum-Labor/qverify/tree/main/docs/architecture.md) —
  detailed walk-through of all four pipeline components

## Citation

```bibtex
@misc{brinza2026qverify,
  author       = {Serghei Brinza},
  title        = {QVerify: quantum-assisted verification of LLM reasoning},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Quantum-Labor/qverify}},
}
```

## Author

[@SergheiBrinza](https://github.com/SergheiBrinza)
