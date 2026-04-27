<div align="center">

# QVerify

### Quantum-Verified Reasoning for Thinking-Mode LLMs

<!-- hero GIF added in Phase 8 -->

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/Quantum-Labor/qverify/actions/workflows/tests.yml/badge.svg)](https://github.com/Quantum-Labor/qverify/actions/workflows/tests.yml)

</div>

---

> **Independent community research project.** Not affiliated with, endorsed by, or sponsored by Google LLC or IBM Corporation. Gemma is a trademark of Google LLC. Built with Gemma — used under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). IBM Quantum is a trademark of IBM Corporation.

## What it does

QVerify intercepts each step of a thinking-mode LLM's reasoning chain, translates that step into a Boolean satisfiability problem in conjunctive normal form, and runs Grover's search on a quantum simulator or real quantum hardware to look for an assignment that satisfies the negation — i.e., a contradiction. When one is found, the controller feeds the result back to the reasoner and asks it to rewrite the step before continuing.

The result: thinking-mode LLMs that catch their own logical errors, verified by quantum search.

## Why this matters

<!-- TODO: filled in Phase 8 with concrete benchmark numbers. -->

## Quick start

```bash
git clone https://github.com/Quantum-Labor/qverify.git
cd qverify
pip install -e ".[dev]"
cp .env.example .env
# edit .env with your tokens
pytest -v
```

## How it works

See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram and component breakdown.

## Benchmarks

Benchmark results will be published in Phase 6. See `docs/benchmarks.md` (coming) for methodology.

## Models

QVerify uses three Gemma 4 instruct models, all centralized in [`qverify/utils/models.py`](qverify/utils/models.py). All three are gated on Hugging Face — accept the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) on each model page before first download.

| Role | Model id | Notes |
|---|---|---|
| Translator | `google/gemma-4-E2B-it` | Small; called once per reasoning step to produce CNF |
| Reasoner (baseline) | `google/gemma-4-E4B-it` | Default thinking-mode model; fits on a single 24 GB GPU in bf16 |
| Reasoner (show-off) | `google/gemma-4-26B-A4B-it` | MoE model used for headline benchmark numbers; multi-GPU |

## Hardware requirements

- Python 3.11+
- 1× NVIDIA GPU with 12 GB+ VRAM for Gemma 4 E2B / E4B inference
- 3× GPU with 24 GB+ VRAM each for Gemma 4 26B MoE benchmarks (optional)
- Free IBM Quantum Open Plan account for hardware runs (optional — the PennyLane simulator works without one)

## Citation

```bibtex
@software{qverify2026,
  author  = {Brinza, Serghei},
  title   = {QVerify: Quantum-Verified Reasoning for Thinking-Mode LLMs},
  year    = {2026},
  url     = {https://github.com/Quantum-Labor/qverify},
  version = {0.1.0}
}
```

## Part of the Quantum Co-Processor program

- **QVerify** — quantum verification of LLM reasoning (this repo)
- **QAgent** — quantum optimization for LLM tool selection (coming soon)
- **QRoute** — quantum routing for Mixture-of-Experts models (coming soon)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
