"""CNF formula representation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Literal(BaseModel):
    """A literal: a predicate applied to terms, optionally negated."""

    predicate: str = Field(..., min_length=1)
    args: tuple[str, ...] = Field(default_factory=tuple)
    negated: bool = False

    model_config = {"frozen": True}

    @field_validator("predicate")
    @classmethod
    def _predicate_format(cls, v: str) -> str:
        if not v[0].isupper() or not v.replace("_", "").isalnum():
            raise ValueError(f"predicate must be CamelCase or snake_case alphanumeric, got: {v!r}")
        return v

    def __str__(self) -> str:
        prefix = "¬" if self.negated else ""
        if not self.args:
            return f"{prefix}{self.predicate}"
        return f"{prefix}{self.predicate}({', '.join(self.args)})"

    def negate(self) -> Literal:
        """Return a new Literal with the negation flag flipped."""
        return Literal(predicate=self.predicate, args=self.args, negated=not self.negated)


class Clause(BaseModel):
    """A disjunction of literals."""

    literals: tuple[Literal, ...]

    model_config = {"frozen": True}

    @field_validator("literals")
    @classmethod
    def _non_empty(cls, v: tuple[Literal, ...]) -> tuple[Literal, ...]:
        if not v:
            raise ValueError("clause must contain at least one literal")
        return v

    def __str__(self) -> str:
        return "(" + " ∨ ".join(str(lit) for lit in self.literals) + ")"


class CNF(BaseModel):
    """A conjunction of clauses (CNF formula)."""

    clauses: tuple[Clause, ...]

    model_config = {"frozen": True}

    def __str__(self) -> str:
        if not self.clauses:
            return "⊤"
        return " ∧ ".join(str(c) for c in self.clauses)

    @property
    def variables(self) -> frozenset[str]:
        """Return the set of distinct predicate symbols in the formula."""
        return frozenset(lit.predicate for cl in self.clauses for lit in cl.literals)

    def to_dimacs(self, var_map: dict[str, int] | None = None) -> str:
        """Render as DIMACS CNF text.

        ``var_map`` assigns positive 1-indexed integer ids to ground atoms,
        keyed by the canonical atom string ``"Predicate(arg1,arg2)"`` (or just
        ``"Predicate"`` for nullary atoms). When ``None``, an id is assigned to
        each distinct atom in lexicographic order.

        Atoms with the same predicate but different argument tuples are
        treated as distinct DIMACS variables, so for Phase 2 we do not
        distinguish between free first-order variables (``Bird(x)``) and
        ground constants (``Bird(penguin)``) — both are stringified into the
        atom key and assigned their own DIMACS id.
        """
        atoms: list[str] = []
        seen: set[str] = set()
        for clause in self.clauses:
            for lit in clause.literals:
                key = _atom_key(lit)
                if key not in seen:
                    seen.add(key)
                    atoms.append(key)

        if var_map is None:
            var_map = {atom: idx + 1 for idx, atom in enumerate(sorted(atoms))}
        else:
            missing = [a for a in atoms if a not in var_map]
            if missing:
                raise ValueError(f"var_map missing entries for atoms: {missing}")
            for atom, vid in var_map.items():
                if vid <= 0:
                    raise ValueError(
                        f"var_map ids must be positive 1-indexed integers; {atom!r} maps to {vid}"
                    )

        nvars = max(var_map.values()) if var_map else 0
        nclauses = len(self.clauses)

        lines: list[str] = [f"p cnf {nvars} {nclauses}"]
        for clause in self.clauses:
            tokens: list[str] = []
            for lit in clause.literals:
                vid = var_map[_atom_key(lit)]
                tokens.append(f"-{vid}" if lit.negated else f"{vid}")
            tokens.append("0")
            lines.append(" ".join(tokens))

        return "\n".join(lines) + "\n"


def _atom_key(lit: Literal) -> str:
    """Canonical string used as a DIMACS variable key for a literal's atom."""
    if not lit.args:
        return lit.predicate
    return f"{lit.predicate}({','.join(lit.args)})"
