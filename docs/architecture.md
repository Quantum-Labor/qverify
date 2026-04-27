# Architecture

QVerify intercepts each step of a thinking-mode LLM's chain of thought, translates the step into a Boolean satisfiability problem in conjunctive normal form, and runs Grover's search on a quantum simulator or real quantum hardware to detect logical contradictions. When the verifier finds a contradiction, the controller feeds the result back to the reasoner and asks it to rewrite the step. The loop continues until the reasoner produces a step that the verifier accepts, then proceeds to the next step.

## Data flow

```mermaid
flowchart TD
    A[User Question] --> B[Reasoner LLM<br/>Gemma 4 E4B / 26B MoE]
    B --> C[Step Interceptor]
    C --> D[Translator LLM<br/>Gemma 4 E2B]
    D --> E[CNF Formula]
    E --> F[Grover Verifier<br/>PennyLane simulator OR IBM Heron r2]
    F --> G{Contradiction<br/>found?}
    G -- Yes --> H[Feedback to Reasoner]
    H --> C
    G -- No --> I{More steps?}
    I -- Yes --> B
    I -- No --> J[Final Verified Answer]
```

## Components

<!-- Filled in Phase 2-5 as each module is implemented. -->

## Quantum verification

<!-- Filled in Phase 3 after the Grover oracle is built. -->

## Hardware backends

<!-- Filled in Phase 4 after IBM Quantum integration. -->
