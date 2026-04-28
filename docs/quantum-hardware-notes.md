# IBM Quantum hardware notes

The Phase 3 verifier ships with two interchangeable backends: a fast PennyLane simulator (the default, used for development and CI) and an IBM Quantum hardware backend that runs the same Grover circuit on a Heron r2 device (default `ibm_kingston`). The simulator stays the default everywhere — hardware is opt-in by passing `backend=IBMQuantumBackend()` to `verify()`.

## Verified hardware runs

| Date       | Backend   | Job ID         | Test                              |
|------------|-----------|----------------|-----------------------------------|
| 2026-04-28 | ibm_fez   | [d7o7dsqk4prs73dt4s6g](https://quantum.cloud.ibm.com/workloads?search=d7o7dsqk4prs73dt4s6g) | test_simple_sat_on_real_hardware  |

## Free IBM Quantum account

1. Sign up at [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com). Open Plan accounts are free and grant roughly 10 minutes of quantum time per month — plenty for hundreds of small Grover runs.
2. From the dashboard, copy your **API key** (the long token at the top of the *API token* card).
3. Copy your **instance CRN** — the cloud-resource name shown next to your instance under *Instances*. It looks like `crn:v1:bluemix:public:quantum-computing:us-east:a/...::`.

## `.env` layout

```
IBM_QUANTUM_TOKEN=<your API key>
IBM_QUANTUM_INSTANCE=<your CRN>
```

Both are optional for the simulator path; the verifier raises a clear `RuntimeError` only when you actually try to run on hardware without them.

## Running the smoke test

```bash
.venv/bin/pytest tests/test_verifier_ibm_smoke.py -v -m slow
```

A successful run:
- prints the resolved backend name (e.g. `IBM job ran on: ibm_kingston`),
- prints the top measurement bitstrings,
- and asserts that the verifier found a counter-model with `Q=True`.

If `IBM_QUANTUM_TOKEN` or `IBM_QUANTUM_INSTANCE` is missing, the test is skipped (it does not fail). On an Open Plan account, expect 0–5 minutes of queue time per submission; the actual quantum job usually executes in well under a second.

## What changes on real hardware

- **Real measurements are noisy.** Grover's amplitude on the marked subspace is finite even on a perfect machine; on a noisy device the histogram is smeared out further by gate errors and decoherence. The classical post-check that already lives in `qverify/verifier/grover.py` (walk `Counter.most_common()` until a satisfying bitstring shows up) is the same hedge that protects the simulator from over-iteration noise — it carries over without change to hardware.
- **Seed does not give you determinism.** `IBMQuantumBackend.execute_grover` accepts `seed` for API parity with `PennyLaneBackend`, but hardware is non-deterministic by construction, so consecutive calls will differ even with identical inputs.
- **Transpilation matters.** We use `optimization_level=3` by default, which routinely cuts gate count by 30–50% versus level 1 on Heron devices. Override via `IBMQuantumBackend(optimization_level=...)` if you need to compare.
- **Backend selection.** With no `backend_name` argument, the backend resolves to the least-busy operational Heron-class device with at least 5 qubits at job-submission time. Pin a specific device with `IBMQuantumBackend(backend_name="ibm_kingston")` if you want reproducible target hardware.

## Phase 4 limits

- The simulator stays the default for `verify()`. Hardware is strictly opt-in.
- We support Heron-class backends only; older Eagle r3 devices are not in the filter.
- Job submission blocks until completion. Asynchronous job tracking lands in a later phase if needed.
