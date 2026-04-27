"""Unit tests for the atom-to-qubit encoder."""

from __future__ import annotations

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier.encoding import AtomEncoder, VerifierError


def _atom(pred: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=pred, args=args, negated=neg)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


def _cnf(*clauses: Clause) -> CNF:
    return CNF(clauses=clauses)


# ---------------------------------------------------------------------------
# atom_names ordering and stability
# ---------------------------------------------------------------------------


def test_atom_names_sorted_lexicographically() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin")), _clause(_atom("Animal", "penguin")))
    enc = AtomEncoder(cnf)
    assert enc.atom_names == ("Animal(penguin)", "Bird(penguin)")


def test_atom_names_stable_across_two_constructions() -> None:
    cnf = _cnf(
        _clause(_atom("Mammal", "felix"), _atom("Cat", "felix")),
        _clause(_atom("Bird", "tweety")),
    )
    e1 = AtomEncoder(cnf)
    e2 = AtomEncoder(cnf)
    assert e1.atom_names == e2.atom_names


def test_atom_names_returns_tuple_immutable() -> None:
    cnf = _cnf(_clause(_atom("P")))
    enc = AtomEncoder(cnf)
    assert isinstance(enc.atom_names, tuple)


def test_n_qubits_matches_unique_atom_count() -> None:
    cnf = _cnf(
        _clause(_atom("Bird", "penguin"), _atom("Bird", "penguin", neg=True)),
        _clause(_atom("Flies", "penguin")),
    )
    enc = AtomEncoder(cnf)
    assert enc.n_qubits == 2  # Bird(penguin), Flies(penguin)


def test_distinct_arg_tuples_produce_distinct_atoms() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin")), _clause(_atom("Bird", "tweety")))
    enc = AtomEncoder(cnf)
    assert enc.n_qubits == 2
    assert enc.atom_names == ("Bird(penguin)", "Bird(tweety)")


def test_predicate_with_no_args_canonical_form() -> None:
    cnf = _cnf(_clause(_atom("Rain")))
    enc = AtomEncoder(cnf)
    assert enc.atom_names == ("Rain",)


def test_empty_cnf_has_zero_qubits() -> None:
    enc = AtomEncoder(CNF(clauses=()))
    assert enc.n_qubits == 0
    assert enc.atom_names == ()


# ---------------------------------------------------------------------------
# atom_to_qubit
# ---------------------------------------------------------------------------


def test_atom_to_qubit_returns_index_in_atom_names() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin"), _atom("Animal", "penguin")))
    enc = AtomEncoder(cnf)
    for i, name in enumerate(enc.atom_names):
        assert enc.atom_to_qubit(name) == i


def test_atom_to_qubit_unknown_atom_raises() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin")))
    enc = AtomEncoder(cnf)
    with pytest.raises(KeyError, match="unknown atom"):
        enc.atom_to_qubit("Cat(felix)")


# ---------------------------------------------------------------------------
# bitstring_to_assignment
# ---------------------------------------------------------------------------


def test_bitstring_to_assignment_msb_first() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin"), _atom("Animal", "penguin")))
    enc = AtomEncoder(cnf)
    # atom_names == ("Animal(penguin)", "Bird(penguin)")
    asn = enc.bitstring_to_assignment("10")
    assert asn == {"Animal(penguin)": True, "Bird(penguin)": False}


def test_bitstring_to_assignment_all_zero() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q"), _atom("R")))
    enc = AtomEncoder(cnf)
    asn = enc.bitstring_to_assignment("000")
    assert asn == {"P": False, "Q": False, "R": False}


def test_bitstring_to_assignment_all_one() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q"), _atom("R")))
    enc = AtomEncoder(cnf)
    asn = enc.bitstring_to_assignment("111")
    assert asn == {"P": True, "Q": True, "R": True}


def test_bitstring_to_assignment_round_trip_via_atom_order() -> None:
    cnf = _cnf(_clause(_atom("Banana"), _atom("Apple"), _atom("Cherry")))
    enc = AtomEncoder(cnf)
    # atom_names sorted: ("Apple", "Banana", "Cherry")
    asn = enc.bitstring_to_assignment("101")
    assert asn[enc.atom_names[0]] is True
    assert asn[enc.atom_names[1]] is False
    assert asn[enc.atom_names[2]] is True


def test_bitstring_to_assignment_wrong_length_raises() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q")))
    enc = AtomEncoder(cnf)
    with pytest.raises(ValueError, match="length"):
        enc.bitstring_to_assignment("111")


def test_bitstring_to_assignment_invalid_chars_raises() -> None:
    cnf = _cnf(_clause(_atom("P")))
    enc = AtomEncoder(cnf)
    with pytest.raises(ValueError, match="0' and '1"):
        enc.bitstring_to_assignment("2")


# ---------------------------------------------------------------------------
# encode_clauses
# ---------------------------------------------------------------------------


def test_encode_clauses_positive_literal_is_polarity_true() -> None:
    cnf = _cnf(_clause(_atom("P")))
    enc = AtomEncoder(cnf)
    assert enc.encode_clauses() == (((0, True),),)


def test_encode_clauses_negated_literal_is_polarity_false() -> None:
    cnf = _cnf(_clause(_atom("P", neg=True)))
    enc = AtomEncoder(cnf)
    assert enc.encode_clauses() == (((0, False),),)


def test_encode_clauses_qubit_indices_match_atom_order() -> None:
    cnf = _cnf(_clause(_atom("Banana"), _atom("Apple", neg=True)))
    enc = AtomEncoder(cnf)
    encoded = enc.encode_clauses()
    # atom_names sorted: ("Apple", "Banana") -> Apple=qubit 0, Banana=qubit 1
    # Original clause had Banana then Apple-neg, so encoded preserves that order
    assert encoded == (((1, True), (0, False)),)


def test_encode_clauses_multiple_clauses() -> None:
    cnf = _cnf(
        _clause(_atom("P")),
        _clause(_atom("Q")),
        _clause(_atom("P", neg=True), _atom("Q")),
    )
    enc = AtomEncoder(cnf)
    encoded = enc.encode_clauses()
    # P=0, Q=1
    assert encoded == (
        ((0, True),),
        ((1, True),),
        ((0, False), (1, True)),
    )


def test_encode_clauses_empty_cnf() -> None:
    enc = AtomEncoder(CNF(clauses=()))
    assert enc.encode_clauses() == ()


def test_encode_clauses_returns_immutable_tuple_of_tuples() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q")))
    enc = AtomEncoder(cnf)
    encoded = enc.encode_clauses()
    assert isinstance(encoded, tuple)
    assert all(isinstance(c, tuple) for c in encoded)
    assert all(isinstance(lit, tuple) for c in encoded for lit in c)


# ---------------------------------------------------------------------------
# Free-variable rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var", ["x", "y", "z", "xs", "abc", "sk1"])
def test_free_variable_in_single_arg_rejected(var: str) -> None:
    cnf = _cnf(_clause(_atom("Bird", var)))
    with pytest.raises(VerifierError, match="free first-order variable"):
        AtomEncoder(cnf)


def test_free_variable_among_multiple_args_rejected() -> None:
    cnf = _cnf(_clause(_atom("Loves", "alice", "y")))
    with pytest.raises(VerifierError, match="free first-order"):
        AtomEncoder(cnf)


def test_free_variable_error_mentions_grounding() -> None:
    cnf = _cnf(_clause(_atom("Bird", "x")))
    with pytest.raises(VerifierError, match="ground"):
        AtomEncoder(cnf)


@pytest.mark.parametrize("constant", ["penguin", "tweety", "alice", "felix", "bobby"])
def test_long_lowercase_constants_accepted(constant: str) -> None:
    cnf = _cnf(_clause(_atom("Bird", constant)))
    AtomEncoder(cnf)  # should not raise


def test_uppercase_constants_accepted() -> None:
    cnf = _cnf(_clause(_atom("Bird", "Penguin")))
    AtomEncoder(cnf)


def test_numeric_token_accepted() -> None:
    cnf = _cnf(_clause(_atom("Equals", "1abc")))
    AtomEncoder(cnf)


def test_zero_arg_predicate_accepted() -> None:
    cnf = _cnf(_clause(_atom("Rain")))
    AtomEncoder(cnf)
