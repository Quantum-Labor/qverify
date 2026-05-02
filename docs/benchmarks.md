# Benchmarks

*Last updated: 2026-05-02 · Status: v1.0 stable*

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

QVerify v1.0 ships **one** benchmark with reportable accuracy and two
NL benchmarks supported by the harness but flagged as out of scope.

| Dataset | Source | License | Status in v1.0 | Notes |
| --- | --- | --- | --- | --- |
| **qverify-mini-50** | hand-crafted (this repo) | Apache 2.0 | **In scope. 100% accuracy.** | 50 examples, 25 SAT / 25 UNSAT, 1-9 atoms. Every label cross-checked against PySAT before commit. See [benchmarks/qverify_mini/README.md](../benchmarks/qverify_mini/README.md). |
| ProofWriter (depth-1) | [Tafjord et al.](https://allenai.org/data/proofwriter) | Apache 2.0 | Out of scope. | Translation works (92/100), but grounded CNFs reach 19-83 atoms — beyond the 16-qubit `MAX_VARIABLES`. |
| RuleTaker (depth-1) | [Clark et al.](https://allenai.org/data/ruletaker) | CC BY 4.0 | Out of scope. | Same atom-count blow-up after grounding (25-54 atoms). Tracked for v1.1 once smarter grounding lands. |

Full attribution and citation details are in
[../benchmarks/LICENSE-DATA.md](../benchmarks/LICENSE-DATA.md).

**FOLIO is excluded from v1.0.** Its CC BY-SA 4.0 license would force
the resulting benchmark report into CC BY-SA, which conflicts with the
project's Apache 2.0 default. `qverify.eval.datasets.load_folio` raises
`NotImplementedError` to make the exclusion explicit at the call site.

Original three-class labels (entails / contradicts / unknown) on
ProofWriter and RuleTaker are mapped to the consistency-mode two-class
label (consistent / inconsistent); examples with the "unknown" gold
label are dropped from the run.

## Headline result — qverify-mini-50

The hand-crafted benchmark exists because the verifier's correctness is
a property of the SAT-checking pipeline, and to measure it honestly we
need (1) CNFs small enough to fit on the simulator, (2) gold labels
that are *not* derived from the verifier itself, (3) coverage of the
failure modes that matter (resolution chains, pigeonhole, modus ponens
contradictions).

| Metric | Value |
| --- | --- |
| Examples | 50 (25 SAT / 25 UNSAT) |
| Accuracy vs PySAT oracle | **100 % (50 / 50)** |
| Avg verify time (simulator) | 3.31 s |
| P95 verify time | 0.32 s |
| Atoms per example | 1 – 9 |
| Skipped | 0 |

Run it:

```bash
python scripts/run_benchmarks.py --dataset qverify-mini --backend simulator \
    --output benchmarks/results/qverify_mini_simulator
```

The full per-example report and PNG charts live at
[benchmarks/results/qverify_mini_simulator/](../benchmarks/results/qverify_mini_simulator/).

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
