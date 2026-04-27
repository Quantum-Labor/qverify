# Translator

The Translator turns one natural-language reasoning step into a Conjunctive Normal Form (CNF) Boolean formula that the Phase 3 verifier can hand to Grover's search. The pipeline is two-stage on purpose: a small instruction-tuned LLM (Gemma 4 E2B) does the open-ended translation, and a strict deterministic parser refuses to accept anything that does not validate against the [`CNF`](../qverify/translator/cnf.py) schema. We never trust the LLM output directly.

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

- `GemmaE2BBackend` — production. Lazy-loads Gemma 4 E2B (gated; default id is `qverify.utils.models.TRANSLATOR_MODEL_ID`, currently `google/gemma-4-E2B-it`) on the first `generate()` call; never at import time. Accept the Gemma license at https://huggingface.co/google/gemma-4-E2B-it before first use. Uses bfloat16 + `device_map="auto"` and greedy decoding for determinism.
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

## Known limitations (Phase 2)

- Propositional and simple first-order only. No higher-order quantification, no modal logic, no arithmetic, no set theory.
- The predicate vocabulary is open-ended — the LLM picks predicate names. There is no closed-world or canonical-name normalization yet, so `Bird` and `IsBird` are different propositions.
- Statements with more than ~3 atomic facts often fail validation, because Gemma 4 E2B has a hard time keeping nested CNF nesting consistent. Phase 6 will measure this; for now, prefer one fact per call.
- DIMACS rendering treats each `Predicate(args)` as a distinct propositional variable. Free first-order variables and ground constants are not distinguished — that grounding is the verifier's job in Phase 3.
- No closed-world assumption: absence of a fact does not imply its negation.
