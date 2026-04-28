"""Controller event types and result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from qverify.verifier.types import CounterModel


@dataclass(frozen=True)
class StreamChunk:
    """One chunk emitted by an :class:`LLMBackend` during streaming."""

    text: str
    phase: Literal["thinking", "answer"]

    def __post_init__(self) -> None:
        if self.phase not in ("thinking", "answer"):
            raise ValueError(f"phase must be 'thinking' or 'answer', got {self.phase!r}")


class ReasoningStepStarted(BaseModel):
    """Emitted when the controller has detected a complete reasoning step."""

    step: str
    timestamp: float

    model_config = {"frozen": True}


class ReasoningStepVerified(BaseModel):
    """Emitted after a verification call returns; success or failure not yet decided."""

    step: str
    attempt: int
    contradiction_found: bool
    timestamp: float

    model_config = {"frozen": True}


class ReasoningStepRejected(BaseModel):
    """Emitted when verification finds a counter-model for the current step."""

    step: str
    counter_model: CounterModel
    attempt: int
    timestamp: float

    model_config = {"frozen": True}


class ReasoningStepCommitted(BaseModel):
    """Emitted when a step passes verification and is appended to the premise list."""

    step: str
    attempt: int
    timestamp: float

    model_config = {"frozen": True}


class ReasoningStepGaveUp(BaseModel):
    """Emitted when all retries fail for a step; the step is dropped, not added to premises."""

    step: str
    last_counter_model: CounterModel
    attempts: int
    timestamp: float

    model_config = {"frozen": True}


class FinalAnswer(BaseModel):
    """Emitted exactly once at the end of streaming with the model's final answer."""

    text: str
    timestamp: float

    model_config = {"frozen": True}


ControllerEvent = (
    ReasoningStepStarted
    | ReasoningStepVerified
    | ReasoningStepRejected
    | ReasoningStepCommitted
    | ReasoningStepGaveUp
    | FinalAnswer
)


class RejectedStep(BaseModel):
    """A reasoning step that was rejected at least once.

    Carries the original step text, the first counter-model produced by
    Grover, the attempt index at which a successor was finally accepted
    (``None`` if the controller gave up), and the final accepted rewrite
    text (``None`` if the controller gave up).
    """

    original_step: str
    counter_model: CounterModel
    fixed_at_attempt: int | None = None
    final_accepted_rewrite: str | None = None

    model_config = {"frozen": True}


class ControllerResult(BaseModel):
    """Full transcript and verification accounting from one reasoning run."""

    final_answer: str
    committed_steps: tuple[str, ...] = Field(default_factory=tuple)
    rejected_steps: tuple[RejectedStep, ...] = Field(default_factory=tuple)
    gave_up_steps: tuple[str, ...] = Field(default_factory=tuple)
    total_verifications: int = 0
    total_contradictions_found: int = 0
    total_groundings: int = 0
    wall_clock_seconds: float = 0.0

    model_config = {"frozen": True}
