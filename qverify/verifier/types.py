"""Result types for the verifier."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class CounterModel(BaseModel):
    """A truth assignment that satisfies an (allegedly contradictory) CNF."""

    assignment: dict[str, bool] = Field(..., description="atom name -> truth value")

    model_config = {"frozen": True}

    def __str__(self) -> str:
        if not self.assignment:
            return "{}"
        parts = [f"{k}={'T' if v else 'F'}" for k, v in sorted(self.assignment.items())]
        return "{" + ", ".join(parts) + "}"


class VerificationResult(BaseModel):
    """Outcome of running Grover's search on a CNF formula."""

    contradiction_found: bool
    counter_model: CounterModel | None = None
    n_variables: int
    n_clauses: int
    n_grover_iterations: int
    backend_name: str
    shots: int
    top_measurements: tuple[tuple[str, int], ...] = Field(
        default_factory=tuple,
        description="Top measurement bitstrings and their counts, descending. Truncated.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-specific metadata (e.g. IBM Quantum job_id). Empty for "
            "the simulator backend except for the backend_name echo."
        ),
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _check_consistency(self) -> VerificationResult:
        if self.contradiction_found and self.counter_model is None:
            raise ValueError("contradiction_found=True requires a counter_model")
        if not self.contradiction_found and self.counter_model is not None:
            raise ValueError("contradiction_found=False forbids a counter_model")
        return self
