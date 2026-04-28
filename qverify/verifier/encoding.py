"""Atom-to-qubit encoding for ground CNF formulas."""

from __future__ import annotations

from qverify.translator.cnf import CNF, Literal
from qverify.verifier._vars import is_free_variable


class VerifierError(RuntimeError):
    """Raised when verification cannot be performed (e.g., free variables, too many qubits)."""


def _canonical_atom_name(lit: Literal) -> str:
    if not lit.args:
        return lit.predicate
    return f"{lit.predicate}({','.join(lit.args)})"


class AtomEncoder:
    """Bidirectional mapping between ground atoms and qubit indices.

    Atoms are sorted lexicographically so the same CNF always produces the
    same encoding. The encoder also enforces the Phase 3 ground-atom rule:
    arguments shorter than 4 characters that start with a lowercase letter
    are treated as free first-order variables and rejected.
    """

    def __init__(self, cnf: CNF) -> None:
        self._validate_no_free_variables(cnf)
        atoms: set[str] = set()
        for clause in cnf.clauses:
            for lit in clause.literals:
                atoms.add(_canonical_atom_name(lit))
        self._atom_names: tuple[str, ...] = tuple(sorted(atoms))
        self._atom_to_qubit: dict[str, int] = {
            name: idx for idx, name in enumerate(self._atom_names)
        }
        self._cnf: CNF = cnf

    @property
    def n_qubits(self) -> int:
        return len(self._atom_names)

    @property
    def atom_names(self) -> tuple[str, ...]:
        """Stable, sorted atom-name tuple. Index = qubit id."""
        return self._atom_names

    def atom_to_qubit(self, atom: str) -> int:
        if atom not in self._atom_to_qubit:
            raise KeyError(f"unknown atom {atom!r}; known atoms: {list(self._atom_names)}")
        return self._atom_to_qubit[atom]

    def bitstring_to_assignment(self, bits: str) -> dict[str, bool]:
        """Convert a measurement bitstring (MSB-first, length n_qubits) to an assignment."""
        if len(bits) != self.n_qubits:
            raise ValueError(
                f"bitstring length {len(bits)} does not match n_qubits {self.n_qubits}"
            )
        if any(c not in "01" for c in bits):
            raise ValueError(f"bitstring may only contain '0' and '1', got {bits!r}")
        return {self._atom_names[i]: bits[i] == "1" for i in range(self.n_qubits)}

    def encode_clauses(self) -> tuple[tuple[tuple[int, bool], ...], ...]:
        """Return clauses as tuples of ``(qubit_index, polarity)`` pairs.

        ``polarity = True`` denotes a positive literal; ``False`` denotes a
        negated literal.
        """
        encoded: list[tuple[tuple[int, bool], ...]] = []
        for clause in self._cnf.clauses:
            encoded.append(
                tuple(
                    (self._atom_to_qubit[_canonical_atom_name(lit)], not lit.negated)
                    for lit in clause.literals
                )
            )
        return tuple(encoded)

    @staticmethod
    def _validate_no_free_variables(cnf: CNF) -> None:
        for clause in cnf.clauses:
            for lit in clause.literals:
                for arg in lit.args:
                    if is_free_variable(arg):
                        raise VerifierError(
                            f"argument {arg!r} in literal {lit} looks like a free "
                            f"first-order variable (lowercase, length < 4). The verifier "
                            f"accepts ground atoms only — call "
                            f"qverify.verifier.grounding.ground_cnf(cnf, universe) first."
                        )
