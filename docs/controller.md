# Controller

*Last updated: 2026-05-02 · Status: v1.0 stable*

The controller is the central feature of QVerify. It streams a thinking-mode LLM (Gemma 4 E4B by default), captures the *thinking* phase for UI display only, then extracts numbered declarative reasoning steps from the *answer* phase — the pattern Gemma 4 uses for its actual conclusions. Each extracted step runs through the existing Translator + Verifier pipeline against the running premise list, and — when Grover's search reports that the step is inconsistent with the premises (UNSAT) — the controller injects a focused correction prompt back into the LLM and asks for a single-sentence rewrite. Each step is committed to the premise list only after it survives verification.

Why answer-phase, not thinking-phase? Gemma 4's thinking phase is freeform meta-commentary ("Let me analyze the premises…", multi-sentence markdown paragraphs); splitting it on `\n\n` produced "steps" the translator's single-sentence contract correctly refused. The answer phase carries the structured numbered reasoning (`1. ... 2. ... 3. ...`) — the right granularity for verification. See [`extract_answer_steps`](../qverify/controller/utils.py).

## The loop

```
        ┌────────────┐
        │   user     │
        │  problem   │
        └─────┬──────┘
              │
              ▼
   ┌───────────────────────┐    StreamChunk(thinking|answer)
   │  Gemma 4 E4B reasoner ├────────────────┐
   │    (thinking mode)    │                │
   └───────▲───────────────┘                ▼
           │                       ┌────────────────────┐
           │  correction           │  paragraph buffer  │
           │  prompt               │     (\n\n split)   │
           │                       └─────────┬──────────┘
           │                                 │ step
           │                                 ▼
           │                    ┌────────────────────────┐
           │                    │  Translator (Gemma E4B)│
           │                    │   premise list + step  │
           │                    └─────────┬──────────────┘
           │                              │ CNF
           │                              ▼
           │                    ┌────────────────────────┐
           │     inconsistency  │  Grover verifier       │
           ├────────────────────│ (PennyLane simulator   │
           │                    │  by default; IBM opt-in)│
           │                    │  mode=consistency       │
           │                    └─────────┬──────────────┘
           │                              │
           │                  ┌───────────┴────────────┐
           │                  │                        │
           │       UNSAT (inconsistent)     SAT (consistent)
           │                  │                        │
           │                  ▼                        ▼
           │   ┌──────────────────────┐    ┌────────────────────┐
           └───┤ ask LLM to rewrite   │    │  commit to premises│
               │ (mini-conversation)  │    │  emit Committed    │
               └──────────────────────┘    └─────────┬──────────┘
                                                     │
                                                     ▼
                                          continue with next step
```

## Public API

```python
from qverify.controller import reason_with_verification

result = reason_with_verification(
    problem="Premises: All cats have fur. Tom is a cat. Question: does Tom have fur?",
)
print(result.final_answer)
print(f"{len(result.committed_steps)} committed, {len(result.rejected_steps)} rejected")
```

For interactive UIs, pass an `emit` callback that receives `ControllerEvent` objects in real time:

```python
events = []
result = reason_with_verification(problem="...", emit=events.append)
```

The seven event types — `ReasoningStepStarted`, `ReasoningStepVerified`, `ReasoningStepRejected`, `ReasoningStepCommitted`, `ReasoningStepGaveUp`, `FinalAnswer` — are all frozen Pydantic models carrying step text, attempt index, and a monotonic timestamp.

## Example trace

For a clean syllogism the event stream looks like:

```
ReasoningStepStarted   (step="All cats have fur.")
ReasoningStepVerified  (attempt=0, contradiction_found=False)
ReasoningStepCommitted (attempt=0)

ReasoningStepStarted   (step="Tom is a cat.")
ReasoningStepVerified  (attempt=0, contradiction_found=False)
ReasoningStepCommitted (attempt=0)

ReasoningStepStarted   (step="Therefore Tom has fur.")
ReasoningStepVerified  (attempt=0, contradiction_found=False)
ReasoningStepCommitted (attempt=0)

FinalAnswer            (text="Yes, Tom has fur.")
```

A run with one rejected step that gets fixed on the first retry adds:

```
ReasoningStepStarted   (step="All birds can fly.")
ReasoningStepVerified  (attempt=0, contradiction_found=True)
ReasoningStepRejected  (attempt=0, counter_model=None)
   --- mini-conversation: "this step is inconsistent because ..." ---
ReasoningStepVerified  (attempt=1, contradiction_found=False)
ReasoningStepCommitted (attempt=1)
```

`counter_model` is `None` because the verifier ran in consistency mode — the rejection is an UNSAT verdict, so there is no satisfying assignment to display. Runs that pass `mode="entailment"` to the verifier directly still surface a witness assignment for SAT-as-rejection cases.

## Why consistency, not entailment?

The earlier (Phase 3) framing checked `premises ∧ ¬step` for satisfiability. SAT meant Grover found an assignment satisfying the premises while making the step false — a counter-example proving the step is *not entailed* by the premises. That works for inference steps but mis-fires on premise-shaped steps: the very first reasoning step is almost always a re-statement of one of the problem's givens (e.g. step 1 = "All cats have fur" on the canonical sanity input). With an empty premise pool, that step is trivially not entailed by anything yet, so entailment-mode verification would always reject step 1 as inconsistent. That's wrong: re-stating a given is not a logical error, it's the bedrock of any chain of reasoning.

Consistency mode (Phase 6.8, the default) checks `premises ∧ step` for satisfiability instead. SAT means the step is *consistent* with the premises so far — accept. UNSAT means the step contradicts the premises — reject. Premise re-statements pose no consistency obstacle (the conjunction is trivially satisfiable when the premise list is empty). Inference steps that *do* follow from the premises are also consistent with them. The mode flips only when the step is a genuine contradiction, which is the case the controller actually needs to catch.

The trade-off: consistency mode can't surface a witness assignment on rejection (UNSAT means there isn't one), so the rewrite prompt falls back to a generic "the step contradicts the established premises" explanation rather than the concrete `{Bird=true, Flies=false}` style breakdown entailment mode produced. Both modes share the same Grover circuit; only the interpretation of the SAT/UNSAT outcome changes — see [`VerifyMode`](../qverify/verifier/grover.py).

## Swapping backends

The controller is dependency-injected at three layers; tests typically swap all three.

| Layer | Production | Tests / CI |
|---|---|---|
| `llm` | `Gemma4ThinkingBackend()` | `StubGemmaBackend(scripts=[...])` |
| `translator` | `Translator(GemmaE2BBackend())` | a translator stub or skipped via `verify_fn` |
| `verifier_backend` | `PennyLaneBackend()` | `verify_fn=StubVerifier([...])` |

For the headline demo, swap only the verifier:

```python
from qverify.verifier.backends import IBMQuantumBackend
result = reason_with_verification(
    problem="...",
    verifier_backend=IBMQuantumBackend(),  # quantum hardware on the wow-button
)
```

The IBM backend is intentionally **not** the inner-loop default — IBM Quantum jobs have queue time and consume Open-Plan minutes; the simulator runs Grover for the same circuit in milliseconds.

## Reproducibility

- Simulator + same `seed` = bit-exact verifier results across runs.
- Same `seed` passed to `Gemma4ThinkingBackend` seeds both `torch.manual_seed` and `transformers.set_seed`, giving deterministic greedy decoding modulo CUDA non-determinism.
- IBM Quantum hardware is intrinsically non-deterministic; `seed` is accepted for API parity but cannot make hardware runs identical.

## v0.1 limits

- The correction prompt is heuristic and fixed. A/B testing alternative phrasings happens in `qverify/controller/correction.py`.
- `max_retries_per_step` is a fixed budget per step. An adaptive budget that grows with reasoning depth is future work.
- Multi-turn conversations are not supported. A single `reason_with_verification` call is one user turn.
- The controller's premise-CNF cache is per-instance. Long sessions that would benefit from a persistent cache are out of scope.
- The Gradio demo (in `space/`) calls the controller synchronously and renders the final result; streaming the `ControllerEvent` stream over SSE / WebSocket is future work.
- `extract_answer_steps` keeps only the first line of each numbered point. Multi-line continuations (a step that wraps onto a second physical line) are dropped past the first line.
- Free-form thinking-phase parsing is intentionally not attempted in v0.1: paragraphs in the thinking phase are multi-sentence and the single-statement translator cannot consume them. The controller therefore verifies the answer phase, which Gemma 4 emits as numbered declarative steps. Free-form thinking-phase reasoning is a v0.2 roadmap item.
