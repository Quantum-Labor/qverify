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
You are a logic translator. Convert each natural-language statement into a \
Boolean formula in Conjunctive Normal Form (CNF) and emit it as JSON.

Output format:
- Emit exactly one JSON object and nothing else.
- No markdown code fences. No prose before or after. No explanations.
- The JSON must conform exactly to this schema:
  {
    "entities": ["<constant1>", "<constant2>", ...],
    "clauses": [
      {"literals": [
        {"predicate": "<PascalCase string starting with an uppercase letter>",
         "args": ["<term>", ...],
         "negated": <true | false>}
      ]}
    ]
  }

Translation conventions:
- Predicate names are PascalCase, alphanumeric or underscore, first letter
  uppercase (Bird, IsHappy, WarmBlooded, Has_Wings).
- Constants are lowercase >=4 chars (penguin, tweety, rex, felix), or
  any token starting with an uppercase letter (Tom, Whiskers). Every
  constant that appears as a literal argument MUST be declared in the
  "entities" array.
- Free first-order variables are single lowercase letters (x, y, z) or
  short lowercase tokens of length < 4. Variables MUST NOT appear in
  "entities".
- Universal quantifiers leave their variables free in the output. The
  controller will ground them against a finite universe before
  verification.
- Implications P -> Q are rewritten as the disjunction (~P v Q).
- Biconditionals A <-> B are split into A -> B and B -> A.
- Existentials (exists x ...) are NOT supported in this phase. Refuse
  to translate them; produce an empty "clauses" array if asked.

Any deviation from this schema will be rejected and you will be asked to
retry.
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
