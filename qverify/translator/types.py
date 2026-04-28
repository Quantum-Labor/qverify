"""Public translator result types."""

from __future__ import annotations

from pydantic import BaseModel

from qverify.translator.cnf import CNF
from qverify.verifier._universe import Universe


class TranslationResult(BaseModel):
    """Output of :meth:`qverify.translator.Translator.translate`.

    Carries the propositional/first-order CNF the LLM produced together
    with the universe of discourse — the constants the controller needs
    to ground the formula before handing it to the verifier.
    """

    cnf: CNF
    universe: Universe

    model_config = {"frozen": True}
