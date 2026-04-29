# Grounding

The Phase 3 verifier accepts only propositional CNF — formulas where every literal is a Boolean variable with no first-order arguments. Real reasoning, on the other hand, almost always emits universally quantified statements: "All cats have fur", "Every bird flies". The Phase 4.5 grounding pass closes that gap by instantiating each free first-order variable with each constant in a finite universe of discourse, producing a propositional CNF the verifier can hand straight to Grover's search.

## The `Universe` model

```python
from qverify.verifier.grounding import Universe

u = Universe(constants=("Tom", "Whiskers"))
```

[`Universe`](../qverify/verifier/_universe.py) is a frozen Pydantic model that wraps a sorted, deduplicated tuple of constant names. Each constant must be a non-empty alphanumeric+underscore string that does not match the free-variable pattern (lowercase with length < 4 — those tokens are reserved for variables like `x`, `y`, `sk1`, etc.). Empty universes are permitted only for purely propositional formulas.

## The `ground_cnf` function

```python
from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier.grounding import ground_cnf, Universe

# forall x. Cat(x) -> Fur(x)
first_order = CNF(clauses=(
    Clause(literals=(
        Literal(predicate="Cat", args=("x",), negated=True),
        Literal(predicate="Fur", args=("x",)),
    )),
))

universe = Universe(constants=("Tom", "Whiskers"))
grounded = ground_cnf(first_order, universe)
```

The expansion of `∀x. Cat(x) → Fur(x)` over `{Tom, Whiskers}` produces:

```
(¬Cat(Tom)      ∨ Fur(Tom))
(¬Cat(Whiskers) ∨ Fur(Whiskers))
```

The whole CNF is treated as one universally quantified formula `∀x.∀y. ... cnf` — the same assignment is applied to every clause simultaneously. For `n` distinct variables and `k` constants, the result has up to `k**n` grounded copies of every clause; identical grounded clauses (e.g. when the same constant is chosen for two variables) are deduplicated. The output ordering is deterministic so benchmarks reproduce bit-exact between runs.

If the input CNF is already propositional, `ground_cnf` returns it unchanged — propositional inputs are zero-cost.

If the CNF has free variables but the universe is empty, `ground_cnf` raises `GroundingError` (a subclass of `VerifierError`) — there are no constants to instantiate against.

## How the controller uses grounding

The Phase 5 [`Controller`](../qverify/controller/controller.py) calls the translator on each premise and on the step itself (translated as-is — no `It is not the case that …` prefix; Phase 6.8 verifies `premises ∧ step` for consistency rather than `premises ∧ ¬step` for entailment), getting back a [`TranslationResult`](../qverify/translator/types.py) (CNF + Universe) per call. It then:

1. Merges the universes — union of every result's `constants`, deduplicated and sorted.
2. Concatenates the CNF clauses across results.
3. Calls `ground_cnf(combined, merged_universe)` to instantiate every free variable.
4. Hands the grounded propositional CNF to `verify(..., mode="consistency")`.

The controller seeds the universe from a one-shot translation of the problem statement before entering the reasoning loop (the Phase 6.5 pre-pass); per-step entities are merged on top.

`ControllerResult.total_groundings` counts how many times step 3 fired during a `reason_with_verification` run, and `ControllerResult.initial_universe_size` reports how many constants the pre-pass extracted — both useful for benchmarks and the demo UI.

## Limitations

- **Finite domains only.** QVerify enumerates the full Cartesian product over the universe; infinite domains are out of scope. Real benchmarks (ProofWriter, RuleTaker, FOLIO) are all finite-domain by design.
- **Naive expansion.** No smart pruning — every variable is grounded against every constant, even when a clause's predicate clearly never references that constant. Optimisation is a future phase if benchmarks demand it.
- **No existentials.** `∃x. ...` is out of scope; the translator is instructed to refuse them.
- **No functions or equality.** Predicates only — no `f(x) = g(y)` or built-in `=`.
- **One flat universe per problem.** Multi-sorted domains (cats, birds, numbers as separate sets) are out of scope.

## Cross-references

- [`docs/architecture.md`](architecture.md) — top-level data flow including the grounding step.
- [`docs/translator.md`](translator.md) — the `entities` JSON field and `TranslationResult` return type.
- [`docs/grover-explained.md`](grover-explained.md) — what happens after grounding hands a propositional CNF to the verifier.
