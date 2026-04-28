"""Universe model — leaf module to avoid import cycles with grounding/encoding."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from qverify.verifier._vars import is_free_variable


class Universe(BaseModel):
    """Finite universe of discourse for grounding first-order CNF formulas.

    Empty universes are permitted only for purely propositional formulas;
    :func:`qverify.verifier.grounding.ground_cnf` raises
    :class:`~qverify.verifier.grounding.GroundingError` when it encounters
    a free variable with no constants to instantiate against.
    """

    constants: tuple[str, ...] = ()

    model_config = {"frozen": True}

    @field_validator("constants")
    @classmethod
    def _validate_constants(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for c in v:
            if not isinstance(c, str):
                raise ValueError(f"constant {c!r} is not a string")
            if not c:
                raise ValueError("constants must be non-empty strings")
            if not c.replace("_", "").isalnum():
                raise ValueError(f"constant {c!r} must be alphanumeric or underscore only")
            if is_free_variable(c):
                raise ValueError(
                    f"constant {c!r} matches the free-variable pattern "
                    f"(lowercase, length < 4); rename it (e.g. 'tom' -> 'Tom' "
                    f"or 'tom_cat')"
                )
            if c in seen:
                raise ValueError(f"duplicate constant {c!r}")
            seen.add(c)
        return tuple(sorted(v))
