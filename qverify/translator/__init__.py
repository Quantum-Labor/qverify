"""Natural language to CNF translation."""

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.translator.translator import TranslationError, Translator

__all__ = ["CNF", "Clause", "Literal", "TranslationError", "Translator"]
