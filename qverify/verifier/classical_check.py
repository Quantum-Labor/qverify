"""Classical CNF satisfaction check."""

from __future__ import annotations

from qverify.translator.cnf import CNF, Literal


def _atom_key(lit: Literal) -> str:
    if not lit.args:
        return lit.predicate
    return f"{lit.predicate}({','.join(lit.args)})"


def satisfies(cnf: CNF, assignment: dict[str, bool]) -> bool:
    """Return True iff ``assignment`` satisfies every clause of ``cnf``.

    Atom names in ``assignment`` must match the canonical form produced by
    :class:`qverify.verifier.encoding.AtomEncoder`. Missing atoms cause
    :class:`KeyError` rather than silent default-to-False.
    """
    return all(_clause_satisfied(clause.literals, assignment) for clause in cnf.clauses)


def _clause_satisfied(literals: tuple[Literal, ...], assignment: dict[str, bool]) -> bool:
    for lit in literals:
        key = _atom_key(lit)
        if key not in assignment:
            raise KeyError(
                f"assignment is missing atom {key!r}; known atoms: {sorted(assignment.keys())}"
            )
        value = assignment[key]
        # positive literal satisfied when value is True;
        # negated literal satisfied when value is False.
        if value != lit.negated:
            return True
    return False
