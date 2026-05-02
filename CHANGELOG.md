# Changelog

All notable changes to QVerify are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## v1.0.1 — 2026-05-02

Hotfix release: the v1.0.0 Hugging Face Space failed to start because
the Dockerfile only `COPY`'d `app.py` from the build context, leaving
`safety.py`, the SVG logo, and the in-repo benchmark reports out of
the image. The Space stage was `RUNTIME_ERROR` until this patch.

### Fixed

- `space/Dockerfile` now copies `safety.py`, `assets/`, and
  `benchmarks/` next to `app.py` so the Space starts cleanly and the
  hero card / benchmarks panel can read the in-repo report files.

### Documentation

- README rewritten with a v1.0 status sheet (release / accuracy /
  test count / hardware / deployment / pipeline / license / position)
  and a sharper "Why this is real engineering" section.
- `docs/benchmarks.md` highlights the qverify-mini-50 result; the
  per-dataset table now lists the in-scope/out-of-scope status per
  benchmark.
- `docs/quantum-hardware-notes.md` "Verified hardware runs" table
  gains the second IBM job (`d7q961poagoc73fj6oag`).
- All `docs/*.md` "Last updated" stamps refreshed to 2026-05-02.

## v1.0.0 — 2026-05-02

First stable release. The verifier has a measurable accuracy number
(100% on a 50-example hand-crafted benchmark with PySAT-validated
labels), the public Space carries the v1.0.0 stable badge, and the
project closes its v1 scope.

### Added

- **qverify-mini-50** hand-crafted benchmark
  (`benchmarks/qverify_mini/dataset.json`): 50 examples, 25 SAT / 25
  UNSAT, 1-9 atoms each (under the 16-qubit simulator ceiling).
  Categories: propositional trivia, modus ponens chains, transitivity,
  resolution, pigeonhole-tiny, grounded first-order, AND/OR mixes.
  Every label was cross-checked against PySAT Glucose3 before commit.
- `qverify.eval.datasets.load_qverify_mini()` loader plus
  `--dataset qverify-mini` in the benchmark CLI. No translator needed;
  every record ships a pre-rendered CNF.
- `benchmarks/results/qverify_mini_simulator/` with report.json and
  three PNG charts. Verified accuracy: **100% (50/50)**, avg
  3.3s per example, P95 0.32s, 0 skipped.
- Space hero card showing the qverify-mini-50 accuracy /
  avg-verify-time / example-count directly under the intro copy,
  sourced from the in-repo report.json.
- Stable-status badge on the Space hero next to the version pill
  ('stable · 433 tests · CI green').

### Changed

- `_coerce_example` accepts records that ship a `rendered_cnf` without
  `premises`/`hypothesis` text; records with neither are still
  rejected.
- README benchmarks section replaced with the qverify-mini-50 numbers
  and a 'Benchmark scope' subsection that names ProofWriter and
  RuleTaker as out-of-scope for v1.0 (atom counts dwarf the 16-qubit
  simulator ceiling).
- Space hero version pill: v0.2.0 -> v1.0.0.

### Test count

433 unit tests (was 432 in v0.2.0), all green in CI.

## v0.2.0 — 2026-05-02

Production-grade public demo. The Hugging Face Space at
[Laborator/qverify](https://huggingface.co/spaces/Laborator/qverify) is
now safe to expose publicly: per-IP rate limit, daily cap, and IBM
quota guard prevent the project's free-tier budget from being drained.

### Added

- **Benchmark harness** (`qverify.eval`): dataset loaders for ProofWriter
  and RuleTaker (FOLIO excluded by license), PySAT Glucose3 oracle,
  metrics models (`DatasetReport`, `ExampleResult`), runner with
  optional `translate` callback and `max_variables` skip filter,
  matplotlib chart renderers, and `scripts/run_benchmarks.py` CLI.
  `scripts/download_datasets.py` lazily fetches each dataset to
  `~/.cache/qverify/datasets/<name>/depth-<N>/<split>.json`.
- **Translator wiring** (`scripts/run_benchmarks.py --translate
  gemma-e2b`): lazily loads Gemma 4 E2B from `qverify.translator`,
  caches grounded CNFs to
  `~/.cache/qverify/translations/<dataset>/depth-<N>/<split>.json`,
  with `--force-translate` to bypass.
- **Hugging Face Space safety** (`space/safety.py` + `space/test_app_safety.py`):
  `RateLimiter` enforces per-IP throttle (1 IBM run / 5 min), global
  daily cap (5 / UTC day), and IBM monthly-quota floor (60 s).
  Persists to `/data/qverify_quota.json` when HF Persistent Storage is
  mounted; degrades silently to in-memory otherwise. 10 unit tests.
- **Quantum Labor visual identity**:
  `assets/quantum_labor_logo.svg` (animated Bloch-sphere wireframe,
  purple #5B21B6 / cyan #06B6D4) embedded in the Space hero.
  Custom Gradio dark theme via `gr.themes.Base` with Inter and
  JetBrains Mono Google Fonts.
- **Visitor-friendly Space copy**: hero block with logo + project
  badge + tagline, four Markdown sections (What is this? / How it
  works / What you'll see / Why this matters), action-first button
  labels, and a "What just happened?" Accordion that explains every
  field of the JSON output in plain English.
- **README polish**: live demo link in the lead paragraph,
  "Why this is real engineering" section with the hardware run /
  test count / license / project-of-3 framing, ASCII pipeline
  diagram, and benchmark table populated with real ProofWriter and
  RuleTaker numbers.

### Changed

- `DatasetReport` adds optional `n_translated`,
  `n_translation_failed`, `avg_translation_seconds`, and
  `n_skipped_too_large` fields with safe defaults.
- `evaluate()` accepts a `max_variables` kwarg; examples whose
  grounded CNF exceeds it are counted in `n_skipped_too_large` and
  bypass `verify()` entirely.
- `qverify.eval.datasets._record_depth` recognizes the RuleTaker
  string `config="depth-N"` field in addition to the ProofWriter
  numeric `QDep` / `depth` fields.
- `_hf_record_to_examples` accepts both the multi-question
  ProofWriter shape (`questions: [...]`) and the singular RuleTaker
  shape (`question` + top-level `label`).
- Translator parser now normalizes lowercase entity names so
  RuleTaker's bare animal/person tokens (`cat`, `cow`, `tom`)
  survive `Universe` validation.
- Space `verify_on_ibm` returns immediately with the IBM Job ID; the
  fallback "Recover a previous job" panel handles WebSocket
  disconnects.
- HF Space `colorTo` switched from `orange` (rejected by HF API) to
  `yellow`.

### Fixed

- Dataset loader path resolver now matches the downloader's
  `<cache>/<dataset>/depth-<N>/<split>.json` layout (was looking
  one level higher).
- Per-dataset split names: ProofWriter uses `validation`,
  RuleTaker uses `dev`. Both are wired through the loader defaults
  and the CLI.
- `tasksource/proofwriter` and `tasksource/ruletaker` are loaded
  with `name="default"`; the per-depth filter is applied
  post-load.

### Verified end-to-end on real quantum hardware

Job [`d7q961poagoc73fj6oag`](https://quantum.ibm.com/jobs/d7q961poagoc73fj6oag)
on `ibm_fez` (IBM Heron r2, 156 qubits), 1024 shots, transpiled depth 360.
Formula `(P ∨ Q) ∧ (¬P ∨ Q)` reported as consistent (3/4 satisfying
assignments found via the classical top-K post-check on the noisy
measurement histogram).

## v0.1.0 — 2026-04-29

Initial public release. Translator (Gemma 4 E2B + outlines), grounding,
verifier (Grover on PennyLane simulator and IBM Quantum hardware), and
controller. 370+ unit tests, hardware smoke test, public Space.
