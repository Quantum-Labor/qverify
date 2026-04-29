# Contributing to QVerify

## Reporting issues

Open a GitHub issue with: a clear title, the smallest reproducer, the output
you expected, the output you got. For verifier output, include the CNF and
backend name.

## Pull requests

Open a draft PR early. Run locally before pushing:

```bash
.venv/bin/pytest -m "not slow and not gpu"
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy qverify
```

CI runs these. They must all pass.

## Test coverage

New code needs tests. The project keeps tests gated by markers:

- default: fast unit tests, run in CI
- `slow`: hardware (IBM Quantum), run manually
- `gpu`: real LLM models, run manually on a GPU host

Tag yours appropriately. Skip on missing env vars cleanly (see the existing
`pytest.skip(...)` patterns in `tests/test_verifier_ibm_smoke.py`).

## Style

Use ruff (formatter + linter) and mypy (strict). The project enforces both
in CI. No exceptions.

GitHub topics (`quantum-computing`, `gemma-4`, `llm`, `verification`,
`qiskit`, `pennylane`, `formal-logic`) are set in the repository settings,
not in code.

## Areas welcoming contributions

See the Roadmap in [README.md](README.md). v0.2 is open: free-form NL
reasoning, existential quantifiers, larger universes. Open an issue to
discuss before starting big work.
