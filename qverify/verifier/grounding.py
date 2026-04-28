"""Finite-domain grounding for first-order CNF formulas."""

from __future__ import annotations

import itertools

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier._universe import Universe
from qverify.verifier._vars import is_free_variable
from qverify.verifier.encoding import VerifierError

__all__ = ["GroundingError", "Universe", "ground_cnf"]


class GroundingError(VerifierError):
    """Raised when grounding cannot be performed (e.g. empty universe)."""


def ground_cnf(cnf: CNF, universe: Universe) -> CNF:
    """Return the propositional CNF obtained by instantiating every free
    first-order variable in ``cnf`` with every constant in ``universe``.

    Treats the entire CNF as one universally quantified formula
    ``forall x. forall y. ... cnf``. For ``n`` variables and ``k`` constants
    the result has up to ``k**n`` grounded copies of each clause; identical
    grounded clauses (e.g. the same constant chosen for two variables) are
    deduplicated. Returns ``cnf`` unchanged when it has no free variables.

    Raises :class:`GroundingError` if ``cnf`` has free variables but
    ``universe`` is empty.
    """
    free_vars = _collect_free_variables(cnf)
    if not free_vars:
        return cnf

    if not universe.constants:
        raise GroundingError(
            f"CNF contains free variables {sorted(free_vars)} but the "
            f"universe has no constants; cannot ground."
        )

    sorted_vars = sorted(free_vars)
    grounded: list[Clause] = []
    seen: set[tuple[tuple[str, tuple[str, ...], bool], ...]] = set()

    for assignment in itertools.product(universe.constants, repeat=len(sorted_vars)):
        substitution = dict(zip(sorted_vars, assignment, strict=True))
        for clause in cnf.clauses:
            new_clause = _substitute_clause(clause, substitution)
            key = tuple((lit.predicate, lit.args, lit.negated) for lit in new_clause.literals)
            if key not in seen:
                seen.add(key)
                grounded.append(new_clause)

    return CNF(clauses=tuple(grounded))


def _collect_free_variables(cnf: CNF) -> frozenset[str]:
    return frozenset(
        arg
        for clause in cnf.clauses
        for lit in clause.literals
        for arg in lit.args
        if is_free_variable(arg)
    )


def _substitute_clause(clause: Clause, substitution: dict[str, str]) -> Clause:
    return Clause(
        literals=tuple(
            Literal(
                predicate=lit.predicate,
                args=tuple(substitution.get(arg, arg) for arg in lit.args),
                negated=lit.negated,
            )
            for lit in clause.literals
        )
    )
