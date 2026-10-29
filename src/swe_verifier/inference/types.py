"""Typed boundaries for the verifier runtime; messages follow Mini-swe-agent's format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Message = dict[str, object]
Role = Literal["policy", "gate", "selector", "trajectory"]
Variant = Literal["untrained", "score_only", "joint_rubric", "two_stage"]


class QueryModel(Protocol):
    def query(self, messages: list[Message]) -> Message: ...


@runtime_checkable
class OutputBudgetModel(QueryModel, Protocol):
    def query_with_output_limit(self, messages: list[Message], max_tokens: int) -> Message: ...


@dataclass(frozen=True)
class Candidate:
    id: str
    message: Message


@dataclass(frozen=True)
class Usage:
    role: Role
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    latency_seconds: float
    cost_usd: float | None


@dataclass(frozen=True)
class Score:
    value: float | None
    outputs: tuple[str, ...]
    rubrics: tuple[str, ...]
    samples: tuple[float | None, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    strategy: Literal["none", "all_step", "selective"]
    candidate_count: int
    threshold: float
    failure_policy: Literal["raise", "first_valid"]

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "all_step", "selective"}:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if self.candidate_count < 1 or not 0 <= self.threshold <= 1:
            raise ValueError("candidate_count must be positive and threshold must lie in [0, 1]")
        if self.failure_policy not in {"raise", "first_valid"}:
            raise ValueError("failure_policy must be raise or first_valid")


@dataclass(frozen=True)
class Budget:
    max_calls: int
    cost_limit_usd: float | None

    def __post_init__(self) -> None:
        if self.max_calls < 1 or (self.cost_limit_usd is not None and self.cost_limit_usd <= 0):
            raise ValueError("Budgets must be positive; use None for an unpriced cost budget")


@dataclass(frozen=True)
class Decision:
    selected: Candidate
    candidates: tuple[Candidate, ...]
    gate_score: Score | None
    scores: tuple[tuple[str, Score], ...]
    triggered: bool
    fallback_reason: str | None
    usage: tuple[Usage, ...]
    selector_outputs: tuple[str, ...]


class InvalidOutput(ValueError):
    """The model output violates the configured verifier contract."""


class BudgetExceeded(RuntimeError):
    """The next model call or selected action would exceed the run budget."""
