# Controller

The controller is the central feature of QVerify. It streams a thinking-mode LLM (Gemma 4 E4B by default), intercepts every completed reasoning step on a paragraph boundary, runs the existing Translator + Verifier pipeline against the running premise list, and — when Grover's search produces a counter-model — injects a focused correction prompt back into the LLM and asks for a rewrite. Each step is committed to the premise list only after it survives verification.

## The loop

```
        ┌────────────┐
        │   user     │
        │  problem   │
        └─────┬──────┘
              │
              ▼
   ┌───────────────────────┐    StreamChunk(thinking|answer)
   │  Gemma 4 E4B (E2B in  ├────────────────┐
   │     thinking mode)    │                │
   └───────▲───────────────┘                ▼
           │                       ┌────────────────────┐
           │  correction           │  paragraph buffer  │
           │  prompt               │     (\n\n split)   │
           │                       └─────────┬──────────┘
           │                                 │ step
           │                                 ▼
           │                    ┌────────────────────────┐
           │                    │  Translator (Gemma E2B)│
           │                    │   premise list + ¬step │
           │                    └─────────┬──────────────┘
           │                              │ CNF
           │                              ▼
           │                    ┌────────────────────────┐
           │     counter-model  │  Grover verifier       │
           ├────────────────────│ (PennyLane simulator   │
           │                    │  by default; IBM opt-in)│
           │                    └─────────┬──────────────┘
           │                              │
           │                  ┌───────────┴────────────┐
           │                  │                        │
           │       contradiction                  no contradiction
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
ReasoningStepRejected  (attempt=0, counter_model={Bird=true, Flies=false})
   --- mini-conversation: "this step is inconsistent because ..." ---
ReasoningStepVerified  (attempt=1, contradiction_found=False)
ReasoningStepCommitted (attempt=1)
```

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

## Phase 5 limits

- The correction prompt is heuristic and fixed; A/B testing alternative phrasings happens in `qverify/controller/correction.py`.
- `max_retries_per_step` is a fixed budget per step; an adaptive budget that grows with reasoning depth is future work.
- Multi-turn conversations are not supported — a single `reason_with_verification` call is one user turn.
- The controller's premise-CNF cache is per-instance; long sessions that would benefit from a persistent cache are out of scope for this phase.
- Streaming the `ControllerEvent` stream over SSE / WebSocket is Phase 7 (Gradio) territory; today the only sink is the `emit` callback.
