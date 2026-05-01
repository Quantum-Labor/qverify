# Benchmarks

*Last updated: 2026-05-01*

## Methodology

The harness in [qverify/eval/](../qverify/eval/) runs the verifier on each
example from a benchmark dataset, checks the verifier's output against the
classical PySAT oracle (Glucose3), and aggregates per-example agreement,
latency, and qubit count into a `DatasetReport`. The verifier runs in
consistency mode (Phase 6.8 default): a satisfiable formula is reported as
`contradiction_found=False`, an unsatisfiable one as `True`. PySAT plays the
role of ground truth.

Examples are loaded from a JSON cache under
`~/.cache/qverify/datasets/<dataset>/<split>.json` or from an explicit path
passed on the CLI. Each record carries `id`, `premises`, `hypothesis`, and
`label` ("consistent" or "inconsistent"). When the record also carries a
`rendered_cnf` (a pre-encoded CNF), the runner skips translation entirely;
this is how Phase 6 numbers are reproducible without a GPU. Records without
a `rendered_cnf` require a `translate` callback (typically the Gemma 4 E4B
translator), which is not exercised in CI.

The runner skips and counts examples that fail to translate, fail the SAT
oracle, or fail the verifier (e.g. ungrounded variables, qubit count above
the simulator's safety cap).

## Datasets

v0.1 ships benchmarks against two datasets:

| Dataset | Source | License | Split used |
| --- | --- | --- | --- |
| ProofWriter | [ProofWriter (Tafjord et al.)](https://allenai.org/data/proofwriter) | Apache 2.0 | depth-1, dev split |
| RuleTaker | [RuleTaker (Clark et al.)](https://allenai.org/data/ruletaker) | CC BY 4.0 | depth-1, dev split |

Full attribution and citation details are in
[../benchmarks/LICENSE-DATA.md](../benchmarks/LICENSE-DATA.md).

**FOLIO is excluded from v0.1.** Its CC BY-SA 4.0 license would force
the resulting benchmark report into CC BY-SA, which conflicts with the
project's Apache 2.0 default. `qverify.eval.datasets.load_folio` raises
`NotImplementedError` to make the exclusion explicit at the call site.

Original three-class labels (entails / contradicts / unknown) are mapped to
the consistency-mode two-class label (consistent / inconsistent); examples
with the "unknown" gold label are dropped from the run.

## Downloading datasets

Real downloads are opt-in via:

```bash
python scripts/download_datasets.py --datasets proofwriter,ruletaker
```

The script writes per-split JSON files to
`~/.cache/qverify/datasets/<name>/depth-<N>/<split>.json` in the schema
the loaders expect. Importing `qverify.eval` triggers no network calls.

## How to reproduce

Place the cached dataset JSON under
`~/.cache/qverify/datasets/<name>/<split>.json` (or pass `--input-path`),
then:

```bash
python scripts/run_benchmarks.py \
    --dataset proofwriter \
    --backend simulator \
    --max-examples 100 \
    --output benchmarks/results/proofwriter_simulator
```

Outputs:

- `benchmarks/results/proofwriter_simulator/report.json` (the full report)
- `benchmarks/results/proofwriter_simulator/accuracy.png`
- `benchmarks/results/proofwriter_simulator/latency.png`
- `benchmarks/results/proofwriter_simulator/qubits.png`

Repeat for `ruletaker` and `folio`. For IBM hardware, swap
`--backend ibm` (consumes Open Plan quantum credit; see
[quantum-hardware-notes.md](quantum-hardware-notes.md)).

## Results

Reports are checked into [benchmarks/results/](../benchmarks/results/) once
generated. The README "Benchmarks" section embeds the chart PNGs directly.
