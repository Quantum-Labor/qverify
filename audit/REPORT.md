# QVerify v1.0 — Audit Report

Read-only audit performed on branch `audit/wow-prep`. No file under `qverify/`,
`space/app.py`, `pyproject.toml`, or `README.md` was modified. All new content
lives under `audit/` and `tests/`.

- Date of audit: 2026-06-06
- Auditor environment: `.venv/bin/python` 3.11.15, pytest 9.0.3, coverage 7.13.5,
  ruff 0.15.12, mypy 1.20.2, bandit 1.9.4, pip-audit 2.10.0, vulture 2.16
- Raw tool output: [`audit/raw/`](raw/) · Profiles: [`audit/profiles/`](profiles/) ·
  Coverage HTML: [`audit/coverage/index.html`](coverage/index.html)

## Assumptions made (no questions were asked)

1. **Coverage / test runs use the CI marker filter** `-m "not slow and not gpu"`.
   The 9 slow/gpu tests require a CUDA GPU, a gated Gemma checkpoint, or live IBM
   credentials and cannot run in this environment. This matches the project's
   documented CI invocation.
2. **Missing audit tools were installed into `.venv`** (bandit, pip-audit,
   vulture, hypothesis, pytest-benchmark, gradio_client). This does not touch
   `pyproject.toml` or any source. `pip-audit` therefore reports against the
   live environment, which includes these added tools.
3. **`mypy --strict qverify`** (the literal command requested) is reported
   separately from the project's own `mypy qverify` config; they give different
   results (see §3).
4. The IBM hardware submit path and the Gemma translator path could not be
   executed (no GPU, no live IBM job allowed in a read-only audit). They were
   audited by static reading and by profiling the offline-measurable portions.

---

## 1. Repository map

Totals: **39 Python modules in `qverify/` = 4,242 LOC**; `space/` = 1,368 LOC;
`scripts/` = 736 LOC; `tests/` = 6,004 LOC (28 files).

### `qverify/verifier/` — Grover search, grounding, encoding

| Module | LOC | Purpose |
| --- | ---: | --- |
| `grover.py` | 172 | `run_grover()` / `verify`: builds the encoder, runs the backend, then a classical top-K rescan; interprets the result per `consistency`/`entailment` mode. `optimal_iterations()` = `round(pi/4 sqrt(N/M))`. `MAX_VARIABLES=16`. |
| `backends.py` | 174 | `Backend` Protocol; `PennyLaneBackend` (statevector `default.qubit`) and `IBMQuantumBackend` (lazy IBM client). |
| `grover_circuit.py` | 139 | `build_grover_qiskit_circuit()` — Qiskit circuit for the IBM path (clause-OR ancillas + flag qubit + diffusion). |
| `oracle.py` | 94 | `build_sat_oracle()` — PennyLane phase-flip SAT oracle, one ancilla per clause + 1 flag; uncomputes ancillas. |
| `encoding.py` | 91 | `AtomEncoder` — bidirectional atom↔qubit map; rejects free variables; `bitstring_to_assignment()`, `encode_clauses()`. |
| `grounding.py` | 78 | `ground_cnf()` — Cartesian-product grounding of first-order CNF over a `Universe`; dedups clauses. |
| `types.py` | 57 | `VerificationResult`, `CounterModel` pydantic models. |
| `diffusion.py` | 48 | `build_diffusion()` — Grover diffusion `H X MCZ X H` + global phase. |
| `_universe.py` | 43 | `Universe` model — validated, sorted, deduped constant tuple. |
| `classical_check.py` | 36 | `satisfies()` — classical CNF satisfaction check over an assignment dict. |
| `_vars.py` | 25 | `is_free_variable()` — heuristic: single alpha char, or lowercase token len<4. |
| `__init__.py` | 26 | Re-exports `verify`, backends, result types, constants. |

### `qverify/translator/` — NL → CNF

| Module | LOC | Purpose |
| --- | ---: | --- |
| `parser.py` | 285 | `parse_llm_output()` — fast path (constrained-gen JSON) + defensive path (fence-strip, balanced-brace extract); entity-casing normalization; entities↔args consistency check. |
| `few_shot.py` | 222 | Few-shot exemplars for the translation prompt. |
| `cnf.py` | 124 | `Literal`/`Clause`/`CNF` pydantic models; `to_dimacs()`; `variables` property. |
| `translator.py` | 118 | `Translator.translate()` — calls a backend, parses, retries up to `max_retries` with error feedback. `TranslationError`. |
| `llm.py` | 112 | `Gemma4StructuredBackend` (outlines constrained gen; alias `GemmaE2BBackend`), `StubBackend`. |
| `schema.py` | 34 | `TranslationSchema` — outlines FSM target schema (entities + clauses). |
| `utils.py` | 25 | `split_sentences()`. |
| `types.py` | 22 | `TranslationResult` (CNF + Universe). |
| `__init__.py` | 14 | Re-exports `CNF`, `Clause`, `Literal`, `Translator`, etc. |

### `qverify/controller/` — orchestration

| Module | LOC | Purpose |
| --- | ---: | --- |
| `controller.py` | 527 | `Controller` / `reason_with_verification()` — problem pre-pass to seed the universe, then per answer-step translate→ground→verify→commit-or-retry loop. |
| `llm.py` | 312 | `LLMBackend` Protocol, `StreamChunk`, `_split_thinking_answer()`, `StubGemmaBackend`, `Gemma4ThinkingBackend`. |
| `types.py` | 147 | Controller event + result models (`ReasoningStep*`, `ControllerResult`, `RejectedStep`). |
| `correction.py` | 61 | `format_counter_model_prompt()` — builds the retry/rewrite prompt. |
| `utils.py` | 61 | `extract_answer_steps()` — pulls numbered steps from the answer phase. |
| `__init__.py` | 41 | Re-exports the controller public API. |

### `qverify/eval/` — benchmark harness

| Module | LOC | Purpose |
| --- | ---: | --- |
| `datasets.py` | 504 | `DatasetExample`, loaders (`load_proofwriter`, `load_ruletaker`, `load_qverify_mini`, `load_folio`), HF download helpers. FOLIO raises `NotImplementedError` (CC BY-SA). |
| `runner.py` | 166 | `evaluate()` — runs the verifier per example, compares vs PySAT oracle, builds report; skip accounting (`n_skipped_too_large`). |
| `metrics.py` | 100 | `ExampleResult`, `DatasetReport`, `build_report()` (accuracy, avg, p95). |
| `charts.py` | 76 | Headless matplotlib renderers (accuracy / latency / qubit-distribution). |
| `oracle.py` | 32 | `pysat_satisfies()` — Glucose3 ground-truth oracle. |
| `__init__.py` | 35 | Re-exports eval API. |

### `qverify/utils/`

| Module | LOC | Purpose |
| --- | ---: | --- |
| `ibm_client.py` | 143 | `IBMRuntimeClient` — lazy `QiskitRuntimeService`, `least_busy_heron()`, `run()` (transpile + SamplerV2, reverses bit order). |
| `config.py` | 47 | `Settings` (pydantic-settings, reads env/.env), `require()`. |
| `models.py` | 25 | Model-ID constants (`TRANSLATOR_MODEL_ID = "google/gemma-4-E4B-it"`). |
| `logging.py` | 22 | `get_logger()`. |
| `__init__.py` | 1 | Docstring only. |

### `space/` — Hugging Face Space (Gradio)

| Module | LOC | Purpose |
| --- | ---: | --- |
| `app.py` | 1,027 | Gradio Blocks UI; simulator + IBM-submit handlers; benchmark/hero cards. |
| `safety.py` | 210 | `RateLimiter` — per-IP window + global daily cap + IBM quota floor; JSON persistence. |
| `test_app_safety.py` | 131 | 10 standalone tests for `RateLimiter` (not in default `testpaths`). |

### `scripts/`

| Module | LOC | Purpose |
| --- | ---: | --- |
| `run_benchmarks.py` | 327 | CLI benchmark driver (dataset/backend/translate/output). |
| `_gen_qverify_mini.py` | 255 | Generator for the qverify-mini-50 dataset, PySAT-validated. |
| `dump_gemma4_thinking.py` | 85 | Diagnostic dump of Gemma thinking-mode tokens. |
| `download_datasets.py` | 69 | Download + reshape ProofWriter/RuleTaker. |

---

## 2. Coverage

Command (CI marker filter — see assumption 1):

```
pytest -m "not slow and not gpu" --cov=qverify --cov-branch \
  --cov-report=html:audit/coverage --cov-report=term-missing
```

Result: **433 passed, 9 deselected. TOTAL coverage 91 %** (1,693 stmts, 129 miss;
514 branches, 50 partial). Full term output: [`audit/raw/coverage_term.txt`](raw/coverage_term.txt);
JSON: [`audit/raw/coverage.json`](raw/coverage.json); HTML: [`audit/coverage/index.html`](coverage/index.html).

### Per-module coverage (lowest first)

| Module | Cover | Notable uncovered lines / branches |
| --- | ---: | --- |
| `translator/llm.py` | 41 % | 45–73, 84–91, 110–111 — real Gemma model load + outlines compile (needs GPU). |
| `utils/ibm_client.py` | 60 % | 94–127 (`run()`), 139–140 — live IBM network path. |
| `utils/config.py` | 71 % | 34–41 (`require()` error branch), 47. |
| `controller/llm.py` | 80 % | 248–261, 270–312 — `Gemma4ThinkingBackend` load/stream (needs GPU). |
| `controller/utils.py` | 89 % | 55, 59→52 — one inline-label branch. |
| `translator/parser.py` | 89 % | 84–85, 92–93, 149, 203–204, 232, 260–261, 263–264, 276 — several defensive error branches. |
| `verifier/backends.py` | 90 % | 136–138 — `IBMQuantumBackend._ensure_client()` settings path. |
| `verifier/oracle.py` | 90 % | 32 (ancilla-mismatch raise), 46 (empty-CNF phase flip). |
| `eval/datasets.py` | 92 % | 391–392, 422–423 — HF download error branches. |
| `eval/runner.py` | 94 % | 133–136 — verify-failure skip branch. |
| `controller/controller.py` | 96 % | 402–403, 502, 518–520 — rewrite/default-backend branches. |
| everything else | 97–100 % | — |

**Uncovered branch themes.** The uncovered code is concentrated in (a) real
model/network paths that cannot run without GPU/IBM, and (b) defensive
error-handling branches in the parser and dataset loaders. The pure logic core
(verifier oracle/diffusion/encoding/grounding/grover, translator cnf, controller
types, eval metrics) is at or near 100 %.

---

## 3. Static analysis (ruff / mypy / bandit / pip-audit)

Raw outputs under [`audit/raw/`](raw/).

### ruff — clean
`ruff check .` → **All checks passed.** `ruff format --check .` → **73 files
already formatted.** Matches the README claim.
([`ruff.txt`](raw/ruff.txt), [`ruff_format.txt`](raw/ruff_format.txt))

### mypy — clean under project config, 3 errors under `--strict`
- `mypy qverify` (project config, what CI runs): **Success: no issues found in 39
  source files.** ([`mypy_projectconfig.txt`](raw/mypy_projectconfig.txt))
- `mypy --strict qverify` (literal audit request): **3 errors**
  ([`mypy.txt`](raw/mypy.txt)):
  - `verifier/backends.py:71,72` — *untyped decorator makes function "circuit"
    untyped* (`@qml.set_shots` / `@qml.qnode` have no type stubs).
  - `controller/llm.py:255` — *call to untyped function "from_pretrained" of
    "AutoProcessor"*.

  These are `--strict`-only checks (`no-untyped-call`, `untyped-decorator`)
  triggered by third-party libraries (PennyLane, transformers) that ship no
  stubs. The README phrasing "mypy strict, all clean" is accurate **for the
  project's configured mypy**, not for full `--strict`. Documented here, not a
  bug — see §7 doc-accuracy notes.

### bandit — 1 Low, 5 Medium (all High confidence)
[`bandit.txt`](raw/bandit.txt). 3,371 LOC scanned, 0 High severity.

| ID | Sev | Location | Issue |
| --- | --- | --- | --- |
| B615 | Med | `translator/llm.py:63,64` | `from_pretrained()` without `revision=` pin (tokenizer + model). |
| B615 | Med | `controller/llm.py:255,256` | `from_pretrained()` without `revision=` pin (processor + model). |
| B615 | Med | `eval/datasets.py:430` | `load_dataset()` without `revision=` pin. |
| B101 | Low | `eval/datasets.py:438` | `assert` used (stripped under `-O`). |

The B615 findings are a real supply-chain note: model/dataset downloads float to
whatever the Hub serves at call time. Low practical risk here (Gemma is gated and
versioned), but pinning `revision=` would make runs reproducible and bandit-clean.

### pip-audit — 16 advisories across 9 packages (live env)
[`pip_audit.txt`](raw/pip_audit.txt). **None are pinned in `pyproject.toml`** (all
deps use `>=` floors), and most are transitive:

| Package | Advisories | Reaches QVerify via | Fix |
| --- | --- | --- | --- |
| starlette | PYSEC-2026-161 | gradio (Space) | 1.0.1 |
| aiohttp | CVE-2026-34993/-47265 | gradio/datasets | 3.14.0 |
| urllib3 | PYSEC-2026-141/-142 | requests stack | 2.7.0 |
| idna | CVE-2026-45409 | requests stack | 3.15 |
| pyjwt | 4× PYSEC-2026 | qiskit-ibm-runtime | 2.13.0 |
| gitpython | CVE-2026-44244, GHSA | wandb | 3.1.50 |
| diskcache, deep-translator, pip | misc | transitive / tooling | — |

`starlette` and `aiohttp` matter most because they sit under the public Gradio
Space. `deep-translator`/`diskcache` are not QVerify dependencies (pulled in by
unrelated env packages) and can be ignored for this project.

### vulture — dead code
[`vulture_qverify.txt`](raw/vulture_qverify.txt), [`vulture_space.txt`](raw/vulture_space.txt).
The majority of `qverify/` hits are **false positives** (pydantic model fields,
`model_config`, field validators, and public API methods exercised by tests or by
the Space). The real findings are listed in §7. The genuine, high-value dead code
is all in `space/app.py` (see §4 and "Bugs found").

---

## 4. Gradio Space audit (`space/app.py`, `space/safety.py`)

### Cold-start time
Measured by importing `app.py` (builds Blocks, examples, RateLimiter, reads
benchmark JSON) without launching, on this machine:

- **Full `app.py` import: 4.58 s wall.** Dominant imports (fresh-process
  `-X importtime`, cumulative): gradio **3.12 s**, pennylane **1.68 s**,
  qiskit-ibm-runtime **0.73 s**, qiskit **0.27 s**, numpy **0.06 s**.
- HF CPU Basic (2 vCPU) is slower than this box, so real cold-start is realistically
  ~10–30 s. The Dockerfile comment claims "~60–120 s on CPU Basic"
  ([`space/Dockerfile:29`](../space/Dockerfile)); that looks **overstated** versus the
  measured import cost, though first-boot disk I/O on a cold container can add to it.

### Memory footprint
**Peak RSS 326 MiB after import** (no model loaded; the Space is verifier-only by
design — torch/transformers/outlines are deliberately omitted from
`space/requirements.txt`). Comfortably within CPU Basic's budget. There is no model
in memory, so the footprint is essentially gradio + pennylane + qiskit.

### Rate-limit logic correctness (`safety.py`)
The `RateLimiter` is well-structured, lock-guarded, commit-on-allow, and covered by
10 unit tests. Logic reviewed line-by-line:

- **Check order**: quota floor → global daily cap → per-IP window. Correct: an
  exhausted monthly quota blocks everyone; the daily cap is global; the per-IP
  window is last. ([`safety.py:96–147`](../space/safety.py))
- **Daily rollover** is UTC-correct (`now.astimezone(UTC).date()`),
  ([`safety.py:160–166`](../space/safety.py)).
- **Persistence** round-trips `{date,count}` and degrades to in-memory on `OSError`.
- **Quota gate does not consume the daily counter** when it fires (verified by
  `test_quota_under_60s_disables_button`). Good.

Two correctness caveats (documented under "Bugs found", not fixed):
1. **`check_and_register` is commit-on-allow, but the IBM submit happens
   afterward** in `verify_on_ibm` ([`app.py:610–628`](../space/app.py)). If
   `_prepare_and_submit` throws (transient IBM error), the daily counter was
   already incremented — a failed submission still burns one of the 5 daily slots.
2. **Per-IP layer is spoofable** — see §9.

### IP-bucket eviction — **there is none**
`self._last_ip: dict[str, float]` ([`safety.py:74`](../space/safety.py)) records one
entry per visitor IP and is **never evicted**. The daily rollover deliberately
leaves it alone, and the comment "Stale per-IP timestamps fade naturally"
([`safety.py:165`](../space/safety.py)) is **incorrect** — nothing removes them. On a
long-running public Space this is an unbounded (slow) memory growth: one float per
unique IP, forever. Practically small, but it is a real leak and the code comment
misdescribes it. A periodic sweep dropping entries older than `window_seconds`
would bound it.

### Dead IBM-result path / stale docstring (most important Space finding)
The module docstring ([`app.py:9–17`](../space/app.py)) describes the IBM verifier
as "a generator that yields a 'submitted' update… polls the job every 8 seconds,
and yields progress + the final result", plus "a separate 'recover by job ID'
panel". **Neither exists in the current code.** `verify_on_ibm` is a plain
submit-and-return function; the Blocks UI ([`app.py:890–1023`](../space/app.py)) wires
only `btn_sim`→`verify_on_simulator` and `btn_hw`→`verify_on_ibm`. There is **no
job-id textbox and no recover button**.

Consequently the following are **defined but unreachable from the UI** (confirmed by
grep — they only call each other):
`POLL_INTERVAL_SECONDS`, `LIVE_POLL_TIMEOUT_SECONDS`, `IBM_TERMINAL_STATUSES`,
`_decode_counts`, `_build_verification_result`, `_final_payload`, `_lookup_job`,
`check_job_status` (~150 LOC).

Root cause is visible in the untracked **`space/app.py.backup`** (751 LOC): it is the
original implementation that *did* poll (`time.sleep(POLL_INTERVAL_SECONDS)`),
*did* call `_build_verification_result`/`_final_payload`, and *did* wire
`check_job_status` to a "Check status by Job ID" button. The rewrite to
submit-only dropped the UI but left the helpers and the docstring behind.

**User-facing impact**: an IBM hardware run in the Space returns a job ID and an
external dashboard link, but the Space **never shows the contradiction verdict for
the hardware path** — the classical post-check (`_build_verification_result`) is
never invoked from the live UI. The headline "watch Grover hunt for contradictions
on real hardware" is only fully delivered for the simulator. This is the single
strongest motivation for the Phase 2 streaming / auto-poll upgrades.

---

## 5. Critical-path profiling

`cProfile` harness: [`audit/profile_harness.py`](profile_harness.py); summary
[`audit/raw/profile_summary.txt`](raw/profile_summary.txt); `.prof` files in
[`audit/profiles/`](profiles/).

| Path | Profile | Cost | Where the time goes |
| --- | --- | --- | --- |
| CNF parse | `cnf_parse.prof` | **0.054 ms/call** | `_normalize_entity_casing` + **3 pydantic validations per parse** (schema fast-path `model_validate_json`, then `CNF.model_validate`, then `Universe`). Re-validates the same data; negligible at any realistic volume. |
| Oracle build | `oracle_build.prof` | **0.002 ms/call** | Pure closure construction; trivial. The cost is when the closure is *applied* inside the simulator (below). |
| Grover sim (6 atoms / 7 clauses / 14 wires) | `grover_sim_6atoms.prof` | **395 ms** | ~99 % inside the PennyLane qnode statevector execution. This is the real cost center. |
| IBM submit (CPU portion only) | `ibm_submit_cpu.prof` | **7.7 ms/call** | `build_grover_qiskit_circuit`: qiskit `_append_standard_gate`, `mcx`, `x`. The network transpile+submit could not be measured offline. |

### Why the simulator is slow — the real scaling law
The PennyLane simulator allocates **`n_atoms + n_clauses + 1` wires** (one ancilla
per clause + a flag), so statevector cost is **`2^(n_atoms + n_clauses + 1)`**, not
`2^(n_atoms)`. Measured on qverify-mini:

| Example | atoms | clauses | total wires | statevector size | verify time |
| --- | ---: | ---: | ---: | ---: | ---: |
| g09 | 6 | 7 | 14 | 16,384 | 0.32 s |
| e03 | 6 | 9 | 16 | 65,536 | 0.86 s |
| **e04** | **9** | **12** | **22** | **4,194,304** | **148–163 s** |

`e04` alone accounts for essentially the entire "average". The `MAX_VARIABLES = 16`
guard checks **atoms only** ([`grover.py:86`](../qverify/verifier/grover.py)); a CNF
with, say, 16 atoms and 15 clauses would request **32 wires (2^32 amplitudes)** and
exhaust memory while still passing the "≤16 variable" check. See "Bugs found".

---

## 6. README claims vs reality

| Claim (README) | Verified? | Evidence |
| --- | --- | --- |
| "433 unit tests, CI green" | **TRUE** | `pytest -m "not slow and not gpu"` collects **433**; total collected 442; 9 slow/gpu deselected. (`def test_` count is 402 before parametrization.) |
| ruff check / ruff format / mypy clean | **TRUE** (with nuance) | ruff clean; `mypy qverify` clean. `mypy --strict` shows 3 third-party-stub errors (§3). |
| qverify-mini-50 = 100 % (50/50) | **TRUE — reproduced** | Re-ran `evaluate(load_qverify_mini())`: **accuracy 1.0**, n=50, 0 skipped. ([`audit/raw/mini_rerun.json`](raw/mini_rerun.json)) |
| "Avg verify time 3.3 s" | **TRUE but misleading** | Committed report `avg_seconds = 3.3068`; my re-run 2.998 s. Both are ~entirely `e04` (148–163 s / 50 ≈ 3 s). 49/50 examples run < 1 s; **P95 = 0.23–0.32 s** is the honest typical figure. |
| "P95 0.32 s" | **TRUE** | Committed 0.3194 s (= g09); re-run 0.234 s. |
| "Atoms per example 1–9, under the 16-qubit cap" | **TRUE for atoms** | Max atoms = 9 (e04). But true register width reaches 22 wires (§5). |
| 10 standalone safety tests | **TRUE** | `space/test_app_safety.py` has 10 tests (not in default `testpaths`). |

**Documentation inconsistencies** (not code bugs, but they undercut the
"honest engineering" framing — full list in §7): package version is **0.1.0**
(`pyproject.toml:7`, `qverify/__init__.py:3`) while everything user-facing says
v1.0.0/v1.0.1; the README mixes **Gemma 4 E2B vs E4B** for the same component
(the code default is E4B); the README/docs cite **two different IBM job IDs and
dates** for "the" hardware run; the Space lists **FOLIO** as a benchmarked dataset
when it is explicitly excluded.

---

## 7. Dead code, TODO/FIXME, and unhandled-exception inventory

### Dead code (genuine)
- **`space/app.py`** — ~150 LOC orphaned by the submit-only rewrite (see §4):
  `POLL_INTERVAL_SECONDS`, `LIVE_POLL_TIMEOUT_SECONDS`, `IBM_TERMINAL_STATUSES`,
  `_decode_counts`, `_build_verification_result`, `_final_payload`, `_lookup_job`,
  `check_job_status`.
- **`space/app.py.backup`** — a 751-LOC stray backup committed to the working tree
  (untracked). Should not ship.
- **`controller/llm.py:_split_thinking_answer`** — only referenced by tests, not by
  production code (it is a tested helper; keep or promote, but it is not on any live
  path).

### Vulture false positives worth recording
`eval/datasets.py:258` "unreachable code after raise" is the intentional
`yield  # pragma: no cover` that keeps `load_folio` a generator for typing — not a
bug. Most other vulture hits are pydantic fields/validators and public API methods
used via tests or the Space.

### TODO / FIXME / XXX / HACK
**None** in `qverify/`, `space/`, or `scripts/`. The codebase carries no TODO debt
markers.

### Unhandled / broad exception paths
All `except Exception` sites are intentional best-effort fallbacks, but two swallow
errors silently in ways worth noting:
- `app.py:243,292,334` — benchmark/report JSON reads `except Exception: continue`
  (a malformed report is silently skipped from the UI tables).
- `app.py:673` `_lookup_job` — `except Exception: pass` before re-raising the
  original lookup error (acceptable; but this is in dead code anyway).
- `_quota_remaining_seconds` ([`app.py:243`](../space/app.py)) returns the stale
  cached value on any IBM API exception — correct degradation, but a persistent IBM
  outage means the quota gate silently relies on the last good value indefinitely.

Type-ignore / noqa: 3 `# type: ignore` in `qverify/` (all justified third-party
gaps: `controller/llm.py:167`, `eval/datasets.py:163`, `translator/llm.py:71`); no
`# noqa` in `qverify/`.

### Doc-accuracy findings (see §6)
1. **Version drift**: `pyproject.toml`/`__init__.py` = `0.1.0`; README badge =
   v1.0.1; `app.py:808` hero = v1.0.0; `pyproject` still `Development Status ::
   3 - Alpha`.
2. **E2B vs E4B**: default model is `google/gemma-4-E4B-it`
   (`utils/models.py:17`); class is named `GemmaE2BBackend` (alias) but loads E4B;
   README lines 25/64 + `docs/architecture.md` say E2B; README lines 118/320/365 +
   `docs/controller.md` + `app.py` say E4B.
3. **Two hardware job IDs**: `d7o7dsqk4prs73dt4s6g` (README:197, 2026-04-28) vs
   `d7q961poagoc73fj6oag` (README:23/32, 2026-05-01), with two different URL bases
   (`quantum.ibm.com/jobs/` vs `quantum.cloud.ibm.com/workloads`).
4. **FOLIO**: `app.py:991` calls the benchmarks "three logic-reasoning datasets
   (ProofWriter, RuleTaker, FOLIO)" but FOLIO is excluded (`datasets.py:254`
   `NotImplementedError`, CC BY-SA).
5. **Stale "370+ unit tests"** at `README.md:399` (the rest of the README says 433).

---

## 8. Dependency pinning audit (`pyproject.toml`)

**Every dependency uses a `>=` floor; nothing is upper-bounded except
`numpy<2.1`.** This is the root cause of the version-fragility findings.

- **Unpinned (`>=` only)**: torch, torchvision, transformers, accelerate, outlines,
  peft, bitsandbytes, qiskit, qiskit-ibm-runtime, pennylane, gradio, hydra-core,
  wandb, pydantic, pydantic-settings, datasets, and all `[dev]` extras.
- **`gradio>=5.0`** is the most consequential: the installed gradio is **6.13.0**,
  in which `theme=` is a `launch()` argument (`Blocks.__init__` has no `theme`).
  `app.py:1027` correctly calls `demo.launch(theme=_QV_THEME)` — **valid on 6.x**.
  But on gradio 5.x `theme` belonged on `Blocks(...)`, so an unpinned floor makes
  the theme placement version-dependent and could silently drop the custom theme or
  raise on a 5.x build. `space/requirements.txt` also pins only `gradio>=5.0`.
- **`numpy<2.1`** is the only ceiling; fine given qiskit/pennylane constraints.
- **Deprecated/abandoned**: none of the declared deps are deprecated. `wandb` is
  heavy and only used by scripts; it is a `dependencies` entry (not `[dev]`), so it
  is pulled into every install of the core package despite being optional in
  practice.
- **Reproducibility gap**: no lockfile (`requirements.lock` / `uv.lock`) is checked
  in, so `pip install -e .` resolves differently over time. Combined with the
  unpinned model `revision=` (bandit B615), runs are not byte-reproducible.

Recommendation (out of scope to apply this pass): add upper bounds for the
fast-moving UI/runtime deps (`gradio`, `qiskit`, `pennylane`, `transformers`),
ship a lockfile, and pin model revisions.

---

## 9. Security

### IBM token handling — **sound**
- Read from env/`.env` via `Settings` (`config.py`) and from `os.environ` in the
  Space (`app.py:440`). **Never logged or echoed** (confirmed by grep across
  `qverify/` and `space/`; the only log line is the job ID).
- **`.env` is git-ignored and untracked** (`.gitignore:1`; `git ls-files` shows only
  `.env.example`). The local `.env` does contain a real-looking IBM token (44-char
  key + 121-char CRN) — present on disk but **not committed**. No leak. (This audit
  did not print the values.) Recommend the usual hygiene: rotate if ever shared and
  keep using Space Secrets, never a committed env.
- The Space reads credentials from HF Space Secrets at runtime; the Dockerfile does
  not COPY `.env`. Correct.

### Rate-limit bypass vectors
- **Per-IP layer is spoofable.** `_client_ip` takes the **first** token of
  `X-Forwarded-For` (`app.py:219–222`). If HF's proxy *appends* the real client IP
  (rather than replacing the header), a client-supplied `X-Forwarded-For` becomes
  the first entry, so an attacker can present a fresh IP per request and bypass the
  1-run/5-min throttle. **Blast radius is bounded by the global daily cap (5/UTC
  day)**, which is not IP-derived — so defense-in-depth holds, but the per-IP layer
  should not be relied on alone. Safer: take the *last* XFF hop, or trust only the
  proxy-set client host.
- **Failed submits still consume the daily cap** (commit-on-allow before submit;
  §4). A flapping IBM backend could let a few transient errors drain the 5/day
  budget. Consider rollback-on-submit-failure.
- **Simulator path is unthrottled** — acceptable: it is local CPU and cheap, but a
  determined client can pin a CPU Basic core. Low risk.

### Gradio CSRF / exposure posture
- `demo.launch()` runs with **no `auth=`** and (by default) **no `share=`** — a
  public, unauthenticated demo. There is no session/cookie credential to ride, so
  classic CSRF is largely **N/A**; the practical risk is direct API abuse, which is
  what the rate limiter addresses (for IBM) — adequate for a public demo.
- Gradio 6 exposes a programmatic API (`/gradio_api`) by default. The IBM function
  is reachable that way too, but it is the same rate-limited path; `show_api` is not
  disabled (informational).
- No user input reaches a shell, filesystem write (except the `/data` quota file),
  `eval`, or template — inputs are a dropdown label + a radio choice, both validated
  against allow-lists (`app.py:404`, `_coerce_mode`). No injection surface.

---

## Bugs found (documented, not fixed per audit scope)

1. **`MAX_VARIABLES` cap measures atoms, not simulated register width.**
   `grover.py:86` rejects CNFs with > 16 *atoms*, but the simulator allocates
   `n_atoms + n_clauses + 1` wires. A formula with 16 atoms and many clauses
   requests `2^(>16)` amplitudes and will OOM/hang despite passing the check. e04
   already reaches 22 wires. **Severity: medium** (silent performance cliff /
   potential OOM on the public Space). Fix idea: cap on total wires, or warn when
   `n_atoms + n_clauses + 1` exceeds a simulator budget.

2. **Dead IBM-result path + stale docstring in `space/app.py`** (see §4). The
   module docstring describes polling + a recovery panel that no longer exist;
   ~150 LOC (`check_job_status`, `_build_verification_result`, `_final_payload`,
   `_lookup_job`, `_decode_counts`, poll constants) are unreachable; the hardware
   path never shows a verdict in-Space. **Severity: medium** (capability gap +
   misleading docs).

3. **Unbounded `_last_ip` growth / wrong comment** (`safety.py:74,165`). No
   eviction; the "fade naturally" comment is false. **Severity: low** (slow leak).

4. **Per-IP rate limit spoofable via `X-Forwarded-For` first-hop** (`app.py:219`).
   **Severity: low–medium** (mitigated by the global daily cap).

5. **Daily cap consumed on failed IBM submit** (commit-on-allow precedes submit).
   **Severity: low.**

6. **Stray `space/app.py.backup`** committed to the tree. **Severity: low**
   (hygiene).

7. **Documentation inconsistencies** — version drift, E2B/E4B, dual hardware job
   IDs, FOLIO mislabel, stale "370+ tests" (§6/§7). **Severity: low** but directly
   at odds with the project's stated "honest engineering" positioning.

No correctness bug was found in the verifier core: grounding, encoding, oracle,
diffusion, classical post-check, and bit-ordering were read and (for the
multi-argument-predicate key path and the 2-arg verify path) **empirically
exercised** — all correct. The qverify-mini-50 100 % result reproduces.
