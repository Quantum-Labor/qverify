"""Few-shot prompts for natural language to CNF translation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    """One in-context demonstration shown to the translator LLM."""

    statement: str
    cnf_json: str
    explanation: str


EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        statement="The penguin is a bird.",
        cnf_json=(
            '{"entities":["penguin"],'
            '"clauses":[{"literals":['
            '{"predicate":"Bird","args":["penguin"],"negated":false}]}]}'
        ),
        explanation=(
            "Atomic ground proposition: single positive literal; the constant "
            "penguin is declared in entities."
        ),
    ),
    FewShotExample(
        statement="Tweety cannot fly.",
        cnf_json=(
            '{"entities":["tweety"],'
            '"clauses":[{"literals":['
            '{"predicate":"Flies","args":["tweety"],"negated":true}]}]}'
        ),
        explanation="Atomic negation: single negated literal.",
    ),
    FewShotExample(
        statement="All birds can fly.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":[{"literals":['
            '{"predicate":"Bird","args":["x"],"negated":true},'
            '{"predicate":"Flies","args":["x"],"negated":false}]}]}'
        ),
        explanation=(
            "Universal implication forall x. Bird(x) -> Flies(x); rewritten "
            "as the contrapositive disjunction (~Bird(x) v Flies(x)). The "
            "variable x is free; no constants are declared because none "
            "appear in the statement."
        ),
    ),
    FewShotExample(
        statement="Every mammal is warm-blooded.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":[{"literals":['
            '{"predicate":"Mammal","args":["x"],"negated":true},'
            '{"predicate":"WarmBlooded","args":["x"],"negated":false}]}]}'
        ),
        explanation=(
            "Universal implication; hyphenated English compound becomes a "
            "single CamelCase predicate WarmBlooded."
        ),
    ),
    FewShotExample(
        statement="If it rains, the street is wet.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":[{"literals":['
            '{"predicate":"Rain","args":[],"negated":true},'
            '{"predicate":"Wet","args":[],"negated":false}]}]}'
        ),
        explanation=(
            "Pure propositional implication Rain -> Wet rewritten as "
            "(~Rain v Wet). No predicate arguments; no entities."
        ),
    ),
    FewShotExample(
        statement="Felix is a cat and Felix is black.",
        cnf_json=(
            '{"entities":["felix"],'
            '"clauses":['
            '{"literals":[{"predicate":"Cat","args":["felix"],"negated":false}]},'
            '{"literals":[{"predicate":"Black","args":["felix"],"negated":false}]}'
            "]}"
        ),
        explanation="Conjunction of ground atoms splits into two unit clauses.",
    ),
    FewShotExample(
        statement="Tweety is a bird or Tweety is a bat.",
        cnf_json=(
            '{"entities":["tweety"],'
            '"clauses":[{"literals":['
            '{"predicate":"Bird","args":["tweety"],"negated":false},'
            '{"predicate":"Bat","args":["tweety"],"negated":false}]}]}'
        ),
        explanation="Disjunction of ground atoms produces a single two-literal clause.",
    ),
    FewShotExample(
        statement="If Rex is a dog, then Rex is a mammal.",
        cnf_json=(
            '{"entities":["Rex"],'
            '"clauses":[{"literals":['
            '{"predicate":"Dog","args":["Rex"],"negated":true},'
            '{"predicate":"Mammal","args":["Rex"],"negated":false}]}]}'
        ),
        explanation=(
            "Ground conditional P -> Q rewritten as (~P v Q). The constant "
            "'Rex' is capitalized because lowercase 3-letter tokens match "
            "the free-variable pattern; entities must be valid constants."
        ),
    ),
    FewShotExample(
        statement="No fish can fly.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":[{"literals":['
            '{"predicate":"Fish","args":["x"],"negated":true},'
            '{"predicate":"Flies","args":["x"],"negated":true}]}]}'
        ),
        explanation=(
            "Universal negation forall x. Fish(x) -> ~Flies(x) yields the "
            "clause (~Fish(x) v ~Flies(x))."
        ),
    ),
    FewShotExample(
        statement="If x is a bird and x is not a penguin, then x can fly.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":[{"literals":['
            '{"predicate":"Bird","args":["x"],"negated":true},'
            '{"predicate":"Penguin","args":["x"],"negated":false},'
            '{"predicate":"Flies","args":["x"],"negated":false}]}]}'
        ),
        explanation=(
            "Conditional with conjunctive antecedent: (Bird(x) ^ ~Penguin(x)) "
            "-> Flies(x) becomes (~Bird(x) v Penguin(x) v Flies(x)) by "
            "distributing the implication over the conjunction."
        ),
    ),
    FewShotExample(
        statement="All birds have feathers and all birds lay eggs.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":['
            '{"literals":['
            '{"predicate":"Bird","args":["x"],"negated":true},'
            '{"predicate":"HasFeathers","args":["x"],"negated":false}]},'
            '{"literals":['
            '{"predicate":"Bird","args":["x"],"negated":true},'
            '{"predicate":"LaysEggs","args":["x"],"negated":false}]}'
            "]}"
        ),
        explanation=(
            "Conjunction of two universal implications produces two clauses, one per implication."
        ),
    ),
    FewShotExample(
        statement="x is a bachelor if and only if x is unmarried and x is a man.",
        cnf_json=(
            '{"entities":[],'
            '"clauses":['
            '{"literals":['
            '{"predicate":"Bachelor","args":["x"],"negated":true},'
            '{"predicate":"Unmarried","args":["x"],"negated":false}]},'
            '{"literals":['
            '{"predicate":"Bachelor","args":["x"],"negated":true},'
            '{"predicate":"Man","args":["x"],"negated":false}]},'
            '{"literals":['
            '{"predicate":"Unmarried","args":["x"],"negated":true},'
            '{"predicate":"Man","args":["x"],"negated":true},'
            '{"predicate":"Bachelor","args":["x"],"negated":false}]}'
            "]}"
        ),
        explanation=(
            "Biconditional Bachelor(x) <-> (Unmarried(x) ^ Man(x)) expands to "
            "three clauses: the two forward implications give "
            "(~Bachelor(x) v Unmarried(x)) and (~Bachelor(x) v Man(x)); the "
            "reverse implication gives "
            "(~Unmarried(x) v ~Man(x) v Bachelor(x))."
        ),
    ),
)


SYSTEM_PROMPT: str = """\
Convert the input statement into a CNF representation.

The output JSON shape is enforced by the decoder — focus on producing the
right *content*. Conventions:

- "entities": list every proper-noun constant the statement mentions
  (people, places, named objects). Constants are either uppercase-led
  (Tom, Whiskers) or lowercase tokens of >= 4 characters (penguin,
  tweety, felix). Leave the list empty for purely universal statements.
- "clauses": each clause is a disjunction of literals. A literal has a
  PascalCase "predicate", an "args" list, and a "negated" bool.
- "args" elements are either declared entities (as ground constants) or
  short lowercase variables (x, y, z) for universally quantified
  statements. Variables do NOT appear in "entities".
- Implications P -> Q become the clause (~P v Q). Biconditionals split
  into both directions.
- Existentials (exists x ...) are NOT supported; produce empty clauses
  if asked.
"""


def build_prompt(
    statement: str,
    examples: tuple[FewShotExample, ...] = EXAMPLES,
) -> str:
    """Assemble system prompt + in-context examples + the target statement."""
    parts: list[str] = [SYSTEM_PROMPT, ""]
    for ex in examples:
        parts.append(f"Statement: {ex.statement}")
        parts.append(f"CNF: {ex.cnf_json}")
        parts.append("")
    parts.append(f"Statement: {statement}")
    parts.append("CNF:")
    return "\n".join(parts)
