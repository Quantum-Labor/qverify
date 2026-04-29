# Translator

*Last updated: 2026-04-29*

The Translator turns one natural-language reasoning step into a [`TranslationResult`](../qverify/translator/types.py) carrying both a (possibly first-order) CNF and the universe of constants the controller needs to ground that CNF before handing it to Grover's search. The production backend uses **grammar-constrained generation** via [outlines](https://github.com/dottxt-ai/outlines): the LLM's token-level generation is constrained by a Pydantic schema, so syntactically invalid output is mathematically impossible. Combined with the strict deterministic parser this gives two layers of defence — outlines guarantees the JSON shape, and the parser surfaces any *semantic* mismatches (wrong predicate case, free variable declared as a constant) with a clear `TranslationParseError`.

## Public return type

```python
from qverify.translator import Translator, TranslationResult
from qverify.translator.llm import GemmaE2BBackend

translator = Translator(backend=GemmaE2BBackend())
result: TranslationResult = translator.translate("The penguin is a bird.")
result.cnf       # CNF — the propositional / first-order formula
result.universe  # Universe — constants declared in this statement
```

`TranslationResult` is a frozen Pydantic model (`cnf: CNF`, `universe: Universe`). The controller merges universes across translator calls and grounds the combined CNF before verification.

## JSON schema the LLM emits

```json
{
  "entities": ["penguin", "Tom", "Whiskers"],
  "clauses": [
    {"literals": [
      {"predicate": "Bird", "args": ["penguin"], "negated": false}
    ]}
  ]
}
```

- `entities` — the universe of declared constants for this statement. Every literal argument that is not a free variable (lowercase, length < 4) must appear here; conversely, no variable may appear here. Empty `[]` is allowed only for purely propositional statements or universals with no concrete constants.
- `clauses` — disjunctions of literals.

The defensive [parser](../qverify/translator/parser.py) validates the entities-vs-literal-args consistency and raises `TranslationParseError` on any mismatch. When the `entities` field is absent in the LLM output, the parser logs a warning and defaults to an empty universe.

## CNF data model

Three frozen Pydantic models in [`qverify/translator/cnf.py`](../qverify/translator/cnf.py):

- `Literal(predicate: str, args: tuple[str, ...], negated: bool)` — predicate names must start with an uppercase letter and contain only alphanumerics or underscores.
- `Clause(literals: tuple[Literal, ...])` — a non-empty disjunction.
- `CNF(clauses: tuple[Clause, ...])` — a (possibly empty) conjunction of clauses; the empty conjunction renders as `⊤`.

Example: "All birds can fly" becomes the single clause `(¬Bird(x) ∨ Flies(x))`:

```python
from qverify.translator import CNF, Clause, Literal

CNF(clauses=(
    Clause(literals=(
        Literal(predicate="Bird",  args=("x",), negated=True),
        Literal(predicate="Flies", args=("x",)),
    )),
))
```

`CNF.to_dimacs()` emits standard DIMACS CNF text suitable for SAT solvers and Grover oracles.

## Retry strategy

Constrained generation has changed what retries mean. With outlines guarding the token-level shape, the parser's first job — *is this valid JSON in the right schema?* — succeeds by construction. The retry budget is now reserved for **semantic** failures: an outlines-emitted output that's syntactically valid but rejected by the parser's deeper checks (lowercase predicate, constant missing from `entities`, etc.). On a freeform-emitting backend (e.g. a stub that returns garbage) the same retry path covers both syntactic and semantic problems.

The `Translator` retries up to `max_retries` times (default 3) on parse failure. The escalation:

| Attempt | Prompt |
|---------|--------|
| 1 | Original few-shot prompt with the target statement |
| 2 | Same prompt — give the model a fair second chance at greedy decoding |
| 3 | Original prompt + the previous bad output + the parser's error message + an explicit "JSON only" instruction |

After the third failure, `Translator.translate()` raises `TranslationError` with all attempted raw outputs attached on `.attempts` (a list of `(raw_output, parse_error)` tuples) for debugging.

Example trace from a hypothetical session:

```
attempt 1 raw: "Here is the CNF:\n```json\n{\"clauses\": ..."
            -> parser strips fence + prose -> SUCCESS
```

```
attempt 1 raw: "I need more context to translate this..."  -> no JSON object
attempt 2 raw: "{\"clauses\": [...]}"                       -> SUCCESS
```

```
attempt 1 raw: "..."           -> invalid JSON
attempt 2 raw: "{\"foo\": []}" -> schema mismatch
attempt 3 raw: "{\"clauses\": [{\"literals\": [...]}]}"  -> SUCCESS  (stricter prompt with feedback)
```

## Backends

[`qverify/translator/llm.py`](../qverify/translator/llm.py) defines the `TranslationBackend` Protocol. Two implementations ship today:

- `Gemma4StructuredBackend` (alias `GemmaE2BBackend`) — production, constrained-generation via outlines. Lazy-loads Gemma 4 E4B by default (gated; default id is `qverify.utils.models.TRANSLATOR_MODEL_ID`, currently `google/gemma-4-E4B-it`) on the first `generate()` call. Accept the Gemma license at https://huggingface.co/google/gemma-4-E4B-it before first use. Uses bfloat16 + `device_map="auto"`. Outlines compiles [`TranslationSchema`](../qverify/translator/schema.py) into a token-level finite state machine on the first call (~few seconds); subsequent translations are normal generate-token speed. The old `GemmaE2BBackend` name is preserved as an alias so existing imports keep working.

  **E2B vs E4B.** Earlier development targeted Gemma 4 E2B (2B params) for the smaller memory and latency footprint, but E2B failed to encode universal quantification reliably — on `"All cats have fur."` it emitted the constant `Cat` where a free variable was required. E4B (4B params) handles this case correctly under the same constrained-generation grammar. The trade-off is roughly 2-3x slower per generation; the Translator runs O(steps) times per reasoning loop, so wall-clock impact is around 5-15 extra seconds per run. To opt back into E2B, pass `Gemma4StructuredBackend(model_id="google/gemma-4-E2B-it")` explicitly.
- `StubBackend(responses: dict[str, str])` — for tests. Returns canned responses keyed by exact prompt.

To swap in a different backend (e.g. an OpenAI-compatible server, a vLLM endpoint, or a different Gemma variant) implement the two-method Protocol and pass the instance to `Translator(backend=...)`.

## Running the GPU smoke test

The unit-test suite uses the stub backend and runs in milliseconds on CPU. The end-to-end smoke test against the real Gemma model is gated behind the `slow` and `gpu` pytest markers and is excluded from CI:

```bash
pip install -e ".[dev]"
.venv/bin/pytest tests/test_translator_e2b_smoke.py -v -m "slow and gpu"
```

A successful run reports two passing tests in roughly 30–90 seconds on a single 12 GB+ GPU. The first call downloads the model weights (~5 GB) into the local HuggingFace cache; subsequent runs are warm.

If you need to pin to a specific revision or substitute a different Gemma E2B-class instruct model, override the model id when constructing the backend:

```python
from qverify.translator.llm import GemmaE2BBackend

GemmaE2BBackend(model_id="google/gemma-4-E2B-it@<revision>")
```

## Known limitations

- Propositional and simple first-order only. No higher-order quantification, no modal logic, no arithmetic, no set theory.
- The predicate vocabulary is open-ended — the LLM picks predicate names. There is no closed-world or canonical-name normalization yet, so `Bird` and `IsBird` are different propositions.
- Statements with more than ~3 atomic facts often fail validation. For now, prefer one fact per call.
- DIMACS rendering treats each `Predicate(args)` as a distinct propositional variable. Free first-order variables and ground constants are not distinguished — that grounding is the verifier's job.
- No closed-world assumption: absence of a fact does not imply its negation.
