"""JSON schema for translator output — source of truth for both LLM grammar and parser."""

from __future__ import annotations

from pydantic import BaseModel, Field


class _LiteralSchema(BaseModel):
    """One literal in a CNF clause as the LLM emits it."""

    predicate: str
    args: list[str] = Field(default_factory=list)
    negated: bool = False


class _ClauseSchema(BaseModel):
    """One clause in the CNF — a non-empty disjunction of literals."""

    literals: list[_LiteralSchema]


class TranslationSchema(BaseModel):
    """Full LLM output: declared entities plus the CNF clauses.

    This schema is the contract that the constrained-generation backend
    enforces at the token level (via :mod:`outlines`) and the contract
    that :func:`qverify.translator.parser.parse_llm_output` validates
    on the way back. Keeping both ends pinned to one Pydantic class
    means a malformed shape is mathematically impossible from the
    structured backend, and parser tests double as schema tests.
    """

    entities: list[str] = Field(default_factory=list)
    clauses: list[_ClauseSchema]
