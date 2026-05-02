# IBM Quantum hardware notes

*Last updated: 2026-05-02 · Status: v1.0 stable*

The verifier ships with two interchangeable backends: a fast PennyLane simulator (the default, used for development and CI) and an IBM Quantum hardware backend that runs the same Grover circuit on a Heron r2 device (default `ibm_kingston`). The simulator stays the default everywhere — hardware is opt-in by passing `backend=IBMQuantumBackend()` to `verify()`.

## Verified hardware runs

| Date       | Backend   | Job ID         | Test / formula                              |
|------------|-----------|----------------|-----|
| 2026-04-28 | ibm_fez   | [d7o7dsqk4prs73dt4s6g](https://quantum.cloud.ibm.com/workloads?search=d7o7dsqk4prs73dt4s6g) | `test_simple_sat_on_real_hardware` (saturated SAT smoke) |
| 2026-05-01 | ibm_fez   | [d7q961poagoc73fj6oag](https://quantum.ibm.com/jobs/d7q961poagoc73fj6oag) | `(P ∨ Q) ∧ (¬P ∨ Q)` — 3/4 satisfying assignments, classical post-check ranks the satisfier at #2 (see "Saturated formulas" below) |

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

## Limits

- The simulator stays the default for `verify()`. Hardware is strictly opt-in.
- Heron-class backends only; older Eagle r3 devices are not in the filter.
- Job submission blocks until completion. Asynchronous job tracking is out of scope for v0.1.

## Saturated formulas on noisy hardware

Grover's algorithm amplifies the marked subspace by a factor proportional to
`√(N/M)`, where `N = 2^n` is the search space size and `M` is the number of
satisfying assignments. When `M ≥ N/2` (a saturated formula), the amplification
is small or negligible: the algorithm cannot meaningfully concentrate amplitude
on any single bitstring because the satisfying set already covers most of the
search space.

On noisy hardware, gate errors and decoherence further flatten the output
distribution. The combined effect is that the measured histogram for a
saturated formula is near-uniform: the most-frequent bitstring is essentially
random with respect to satisfaction, and reading the answer off the top
measurement alone is unreliable.

The classical post-check in [qverify/verifier/grover.py](../qverify/verifier/grover.py)
walks `Counter.most_common()` until it finds a bitstring that classically
satisfies the CNF. This walk is the safeguard for saturated formulas on
hardware: as long as a satisfier appears anywhere in the measured set, the
verifier classifies the formula correctly regardless of which bitstring
happened to top the histogram.

### Concrete observed example

Job `d7q961poagoc73fj6oag` on `ibm_fez` (2026-05-01) ran the CNF
`(P ∨ Q) ∧ (¬P ∨ Q)`, which has 3 satisfying assignments out of 4 (only
`P=False, Q=False` falsifies it). Top measurements:

| Bitstring | Count |
| --- | --- |
| `00` | 292 |
| `01` | 263 |
| `11` | 254 |
| `10` | 215 |

The histogram is almost uniform, as the saturation analysis predicts. The
most-frequent bitstring `00` is the unique non-satisfier. The classical
post-check skipped it, found a satisfier at the next rank, and the verifier
correctly classified the formula as consistent (`contradiction_found=False`)
in consistency mode.
