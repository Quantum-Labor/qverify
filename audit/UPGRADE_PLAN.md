# QVerify — Wow-Factor Upgrade Plan

Companion to [`REPORT.md`](REPORT.md). Scores every mandatory candidate on
**impact (1–5)** and **effort (hours)**, gives a word-mockup, risk, dependencies,
and the **HF Pro feature** each unlocks, then recommends a ~40-hour sprint.

Paid HF Pro is assumed available: **ZeroGPU** (on-demand A100 slices),
**GPU Spaces**, **persistent storage** (`/data`), and **larger builds**.

Scope note: this plan is design-only. Implementing any item will touch
`space/app.py` and friends, which are frozen for the audit pass; the
prerequisites below (dead-code cleanup, `pysat` in the Space requirements) come
from [`REPORT.md`](REPORT.md).

---

## Scoring summary

Effort is a single-engineer estimate including tests. "Ratio" = impact² / effort
(higher = better return); it is a sort key, not gospel.

| # | Upgrade | Impact | Effort (h) | Ratio | HF Pro feature unlocked |
| ---: | --- | :---: | :---: | :---: | --- |
| 10 | Examples carousel (10 curated CNFs) | 4 | 5 | 3.2 | none (CPU) |
| 9 | Shareable branded result cards (SVG) | 4 | 6 | 2.7 | persistent storage (optional) |
| 4 | Streaming verification (live prob / top-k / bars) | 5 | 12 | 2.1 | none (CPU) |
| 11 | API endpoint docs (`gradio_client`) | 2 | 3 | 1.3 | none |
| 7 | Quantum vs Classical race | 4 | 9 | 1.8 | none (CPU; needs `pysat`) |
| 1 | ZeroGPU Gemma translator (one-click NL→verdict) | 5 | 16 | 1.6 | **ZeroGPU + larger build** |
| 5 | IBM job auto-poll + recovery | 5 | 12 | 2.1 | **persistent storage** |
| 2 | Live SVG Grover circuit animation | 5 | 20 | 1.25 | none (CPU) |
| 6 | Persistent leaderboard | 3 | 12 | 0.75 | **persistent storage / HF Dataset** |
| 8 | Interactive Grover tutorial mode | 3 | 14 | 0.64 | none |
| 3 | 3D Bloch sphere (plotly) | 3 | 10 | 0.9 | none (CPU) — correctness caveat |

A recurring theme: **most of the "wow" is CPU-only and does not strictly require
HF Pro.** Pro is essential for exactly two things — putting the **translator** in
the Space (ZeroGPU) and **durable state** (persistent storage for IBM recovery,
leaderboard, and served cards). Plan accordingly.

---

## Candidate detail

### 1. ZeroGPU Gemma translator — one-click NL → CNF → verify  ·  Impact 5 · 16 h
Move the translator into the Space behind `@spaces.GPU` so a visitor types a
sentence and the Space does translate → ground → verify end to end, instead of
hand-picking a pre-built CNF. This is the single feature that turns the demo from
"for people who already have a CNF" into the actual product story.

- **Mockup (words):** A new top input box "Describe a situation in plain English
  (e.g. *All cats have fur. Tom is a cat. Tom has no fur.*)". A "TRANSLATE & VERIFY"
  button. Below it, three collapsing cards appear in sequence: **(1) Parsed CNF**
  (the rendered clauses + universe), **(2) Grounded CNF** (qubit count), **(3)
  Verdict** (reuses the existing simulator result JSON, ideally streamed — see #4).
  A small "ZeroGPU" chip shows GPU acquire/translate/release timing.
- **HF Pro:** ZeroGPU (the only way to run a 2–4B model in a Space without a paid
  always-on GPU), plus larger build to bundle transformers/outlines.
- **Effort breakdown:** `@spaces.GPU` wiring + lazy load 2h; reuse
  `Translator(Gemma4StructuredBackend(...))` 2h; outlines FSM compile + warmup
  caching 3h; UI + state plumbing 4h; error/timeout handling 2h; tests (stub
  backend path + a gated slow GPU smoke) 3h.
- **Risk:** **High.** ZeroGPU caps per-call wall-time (≈60 s default); the first
  call pays model download + outlines FSM compile (tens of seconds) and can exceed
  the cap. **Use Gemma 4 E2B here, not E4B** — E2B fits the ZeroGPU time/memory
  budget far better; accept the documented universal-quantification quality loss
  for the demo, or fall back to a CPU stub when GPU is unavailable. Gemma is gated
  (license acceptance required for the Space token). Resolve the E2B-vs-E4B naming
  confusion (REPORT §7) before shipping copy.
- **Dependencies:** Gemma license accepted on the Space token; `transformers`,
  `outlines`, `torch`, `accelerate`, `spaces` added to the Space build (currently
  omitted by design). Pin model `revision=` (REPORT §3 B615).

### 2. Live SVG Grover circuit animation  ·  Impact 5 · 20 h
Animate the circuit being built gate-by-gate, with amplitude bars updating per
Grover iteration — the marquee "watch the quantum algorithm work" feature.

- **Mockup (words):** A wide panel renders the circuit as an SVG grid (wires =
  rows, gates = columns). On run, gates fade in left-to-right: Hadamard wall →
  oracle (clause-OR ancillas light up) → diffusion. Beneath it, a bar chart of the
  2ⁿ basis-state amplitudes animates: after each iteration the satisfying state's
  bar grows while the rest shrink, visually showing amplitude amplification. A
  step counter "Iteration k / N" advances.
- **HF Pro:** none required (CPU). Benefits from persistent storage to cache
  per-example snapshot frames.
- **Effort breakdown:** per-iteration statevector capture by running the existing
  `build_sat_oracle`/`build_diffusion` in a Python loop and reading the statevector
  after each iteration (PennyLane `qml.state()` on a no-shots device) 5h; SVG/HTML
  renderer for circuit + bars 8h; Gradio streaming/animation wiring 4h; tests 3h.
- **Risk:** Medium. Only legible for small circuits (≤ ~5 atoms / ≤ ~12 wires);
  above that the bar chart has too many bars and the statevector capture is slow
  (REPORT §5 scaling). Gate it to small examples and say so. Pure-SVG animation in
  `gr.HTML` needs careful state diffing to avoid full-rerenders.
- **Dependencies:** pairs with #4 (shares the per-iteration loop) and #10 (small
  curated examples).

### 3. 3D Bloch sphere (plotly) for ≤4-qubit examples  ·  Impact 3 · 10 h
Per-qubit Bloch spheres synced to the current circuit step.

- **Mockup (words):** A row of up to 4 plotly Bloch spheres, one per atom, each
  showing its qubit's state vector; a step slider scrubs through the circuit and
  the arrows update.
- **HF Pro:** none (CPU, plotly via `gr.Plot`).
- **Effort:** reduced-density-matrix extraction per qubit 4h; plotly Bloch render
  4h; slider sync 2h.
- **Risk:** **Correctness/honesty caveat.** Grover entangles the register; a single
  qubit's reduced state lands *inside* the Bloch ball (mixed), and a product-state
  Bloch picture is **physically misleading** once entanglement appears. If shipped,
  it must render the reduced density matrix (point inside the ball) and label it as
  such, not draw a pure-state arrow. Lower impact than it looks and easy to get
  wrong — recommend deferring or reframing as "single-qubit warmup only".
- **Dependencies:** plotly; the per-step statevector from #2/#4.

### 4. Streaming verification  ·  Impact 5 · 12 h
Turn the simulator handler into a generator that yields iteration count, running
success probability, and the live top-k bitstrings as Grover proceeds.

- **Mockup (words):** After "RUN ON SIMULATOR", the result card updates live:
  a progress line "Iteration 3/6 — P(satisfying) = 0.71", a top-5 bitstring table
  re-sorting each tick, then the final verdict. Feels alive even at sub-second
  runtimes (add a small per-iteration delay for legibility on tiny circuits).
- **HF Pro:** none (Gradio queue/generators work on free tier).
- **Effort:** iteration-by-iteration Grover loop with statevector readout +
  success-probability computation (sum |amp|² over classically-satisfying states)
  5h; generator handler + JSON/table streaming 4h; tests (final value matches the
  one-shot `run_grover`) 3h.
- **Risk:** Low–medium. Must not change verdict semantics — assert the streamed
  final result equals the existing `run_grover` output (regression-lock it). The
  current one-shot qnode can't yield mid-call, so this re-implements the loop in
  Python around the existing oracle/diffusion builders (no change to `qverify/`).
- **Dependencies:** foundation for #2, #7, #8.

### 5. IBM job auto-poll + recovery  ·  Impact 5 · 12 h
Reinstate live polling (queue position, estimated wait, status badge, Workloads
link) **and** the "recover by job ID" panel — directly closing the gap in
REPORT §4 where the hardware path never shows a verdict in-Space.

- **Mockup (words):** After submit, a status badge cycles QUEUED → RUNNING → DONE
  with "queue position 4, est. wait ~90 s", auto-refreshing. On DONE it renders the
  full verdict (contradiction found + counter-model) via the *already-written*
  `_build_verification_result`. A separate "Recover a previous run" box takes a Job
  ID and, using metadata persisted at submit time, reproduces the full verdict.
- **HF Pro:** **persistent storage** — persist `{job_id → prepared-CNF metadata}`
  to `/data` so a verdict is recoverable after a websocket drop (the current dead
  `check_job_status` can only show raw counts precisely because that metadata was
  not persisted).
- **Effort:** revive the polling generator from `app.py.backup` 3h; queue/ETA from
  `job.metrics()`/status 3h; persist+reload prepared-job metadata 3h; tests with a
  mock IBM backend (submit→poll→retrieve, no creds) 3h.
- **Risk:** Medium. HF free-tier websocket idle timeout can still cut long queues —
  persistence is the mitigation. Full validation needs live IBM credentials; build
  against a mock backend (see Phase 3 test added this pass).
- **Dependencies:** **delete the orphaned dead code first** (REPORT bug #2) so the
  revived path is the only one. Pairs with #9 (cards embed the Job ID).

### 6. Persistent leaderboard  ·  Impact 3 · 12 h
Track fastest verify, largest solved CNF, most contradictions caught.

- **Mockup (words):** A "Hall of Fame" tab with three small tables; each verify
  run optionally appends an entry (opt-in, no PII — just a chosen handle + metrics).
- **HF Pro:** **persistent storage** (`/data` JSON) or a private **HF Dataset**
  (durable, versioned, survives Space restarts — preferred).
- **Effort:** storage layer + schema 4h; append/read with the existing lock pattern
  4h; UI 2h; anti-spam (rate-limit writes, cap handle length) 2h.
- **Risk:** Medium. Concurrency/write contention and abuse (fake records). HF
  Dataset writes need a token with write scope. Lower priority — engagement, not
  core story.
- **Dependencies:** reuse `safety.py`'s lock + persistence idiom.

### 7. Quantum vs Classical split-screen race  ·  Impact 4 · 9 h
Same CNF on Grover (simulator, visibly iterating) vs PySAT Glucose3 (instant),
side by side, in real time.

- **Mockup (words):** Two columns. Left "Grover (quantum)" streams iterations and a
  stopwatch; right "Glucose3 (classical)" flips to a result almost immediately. A
  banner states the honest punchline the README already makes: *classical wins
  today; the point is the same `verify()` interface scales to hardware.*
- **HF Pro:** none (CPU). Requires `python-sat` added to `space/requirements.txt`
  (currently `[dev]`-only).
- **Effort:** reuse #4's streaming for the quantum side 2h; wire `pysat_satisfies`
  for the classical side 1h; two-column UI + stopwatch 4h; tests 2h.
- **Risk:** Low. Honest framing matters — do not imply quantum is faster. The
  oracle ancilla scaling (REPORT §5) means the quantum side stalls on big inputs;
  cap to curated small examples.
- **Dependencies:** #4, #10; add `python-sat` to the Space build.

### 8. Interactive Grover tutorial mode  ·  Impact 3 · 14 h
Guided step-through of oracle → diffusion → measurement with plain-English
explanations at each stage.

- **Mockup (words):** A "Learn" tab with Next/Back buttons. Each step shows the
  current sub-circuit, a one-paragraph explanation ("the oracle flips the phase of
  satisfying states…"), and the amplitude bars before/after. Ends on measurement +
  the classical post-check.
- **HF Pro:** none.
- **Effort:** content writing 5h; reuse #2/#4 visuals 4h; step state machine + UI
  3h; tests 2h.
- **Risk:** Low but content-heavy; overlaps #2/#4 (build those first or it
  duplicates work).
- **Dependencies:** #2, #4.

### 9. Shareable branded result cards (SVG)  ·  Impact 4 · 6 h
Auto-generate a branded SVG summarizing a run: logo, CNF, verdict, qubit/iteration
counts, backend, Job ID, timestamp — a one-click social/portfolio artifact.

- **Mockup (words):** After any run, a "Download result card" button produces a
  1200×630 SVG/PNG: Quantum Labor logo top-left, "CONSISTENT ✓ / CONTRADICTION ✗"
  verdict band, the CNF in mono, a footer with backend + Job ID + UTC timestamp.
- **HF Pro:** persistent storage optional (to host a permalink to the card);
  download-only needs nothing.
- **Effort:** SVG template reusing `assets/quantum_labor_logo.svg` + design tokens
  3h; populate from the result dict 1h; `gr.File`/`gr.Image` download 1h; tests
  (valid SVG, fields present) 1h.
- **Risk:** Low. Best impact-per-hour after the carousel; strong marketing value.
- **Dependencies:** none (works for simulator now; richer once #5 supplies Job ID).

### 10. Examples carousel (10 hand-picked CNFs)  ·  Impact 4 · 5 h
Replace the 3 built-in examples with 10 curated cases showcasing edge cases: Horn
clauses, 2-SAT, an UNSAT core, pigeonhole-tiny, transitivity, a grounded universal,
a binary predicate, an empty CNF (⊤), a single contradiction, an AND/OR mix.

- **Mockup (words):** The dropdown becomes a horizontal carousel of cards, each
  with a title, a one-line "what this shows", expected verdict, and qubit count.
  Clicking loads it into the existing CNF/universe boxes.
- **HF Pro:** none.
- **Effort:** author + PySAT-validate 10 CNFs (extend `_gen_qverify_mini.py`) 3h;
  carousel UI reusing `_build_examples`/`_on_example_change` 2h.
- **Risk:** Very low. **Do this first** — it is the content substrate every other
  visual feature (#2/#4/#7/#8) needs, and it directly enriches the existing UI.
- **Dependencies:** none.

### 11. API endpoint documentation (`gradio_client`)  ·  Impact 2 · 3 h
Document programmatic access; Gradio already exposes the API.

- **Mockup (words):** An "API" accordion with a copy-paste `gradio_client` snippet:
  `Client("Laborator/qverify").predict(example, mode, api_name="/verify_on_simulator")`,
  plus the response schema and the IBM rate-limit caveat.
- **HF Pro:** none.
- **Effort:** name the endpoints (`api_name=`) for stable handles 1h; write docs +
  a runnable example 1h; a `gradio_client` smoke test 1h.
- **Risk:** Very low. Cheap credibility win; pairs with the Phase 3 integration
  test added this pass.
- **Dependencies:** none.

---

## Recommended 2-week sprint (~40 h) — "One-click quantum verification"

Goal: deliver the **complete product narrative end-to-end** — type English →
translate on GPU → watch Grover converge live → share the result — which is the
highest-wow, demo-ready story and exercises three HF Pro features.

| Order | Item | h | Why it's in |
| ---: | --- | ---: | --- |
| 1 | **#10 Examples carousel** | 5 | Content substrate for everything else; ships value on day one. |
| 2 | **#4 Streaming verification** | 12 | Core "alive" feel; foundation for the race/tutorial later. |
| 3 | **#1 ZeroGPU translator (E2B)** | 16 | The centerpiece — turns the Space into the real product; the only item that *needs* Pro/ZeroGPU. |
| 4 | **#9 Shareable result cards** | 6 | Viral/portfolio payoff; best impact-per-hour; closes the loop. |
| | Buffer | 1 | — |
| | **Total** | **40** | |

Prerequisite (≈2 h, folds into item 1): **delete the orphaned IBM dead code and
fix the stale `app.py` docstring** (REPORT bug #2), and add `transformers /
outlines / torch / accelerate / spaces` to the Space build for ZeroGPU.

Why not the others this sprint:
- **#5 IBM auto-poll** (impact 5) is the top of **Sprint B** — it fixes a real gap
  and needs persistent storage, but full validation needs live IBM credentials, so
  it is better isolated. Build it on the mock-IBM test added in Phase 3.
- **#2 full SVG animation** (20 h) is too big to co-fund with #1 in one sprint; a
  lightweight amplitude-bar view rides along inside #4, and the full circuit
  animation lands in Sprint B.
- **#3 Bloch** carries a physics-correctness caveat; defer until it can be done
  honestly (reduced density matrices).

### Sprint B backlog (next ~40 h)
#5 IBM auto-poll + recovery (12) · #2 full SVG circuit animation (20) ·
#7 Quantum-vs-Classical race (9) — totals ~41 h, all building on this sprint's
streaming + carousel foundation. Then #6 leaderboard, #8 tutorial, #11 API docs,
#3 Bloch as polish.

---

## Cross-cutting prerequisites (one-time)
1. **Resolve E2B/E4B naming** (REPORT §7) before any translator copy ships.
2. **Add `python-sat`** to `space/requirements.txt` for #7 (and the oracle parity
   tests).
3. **Enable persistent storage** on the Space for #5/#6/#9 permalinks.
4. **Pin model `revision=`** (REPORT §3) so ZeroGPU translations are reproducible.
5. **Upper-bound `gradio`** in the Space requirements (REPORT §8) so streaming and
   `theme=` behavior do not shift under the team.
