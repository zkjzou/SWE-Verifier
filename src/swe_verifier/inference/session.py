"""Account for each policy, gate, and selector call before another call is made."""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping

from swe_verifier.inference.types import Budget, BudgetExceeded, Message, QueryModel, Role, Usage


def _count(mapping: Mapping[str, object], name: str) -> int | None:
    value = mapping.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def response_usage(response: Message, role: Role, model_name: str, elapsed: float) -> Usage:
    extra = response.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("Model response extra must be an object")
    raw = extra.get("usage", {})
    raw_response = extra.get("response", {})
    if not raw and isinstance(raw_response, dict):
        raw = raw_response.get("usage", {})
    if not isinstance(raw, dict):
        raw = {}
    details = raw.get("prompt_tokens_details", {})
    cost = extra.get("cost")
    if cost is not None and (not isinstance(cost, (float, int)) or not math.isfinite(cost) or cost < 0):
        raise ValueError("Model response cost must be a finite nonnegative amount or None")
    return Usage(
        role, model_name, _count(raw, "prompt_tokens"), _count(raw, "completion_tokens"),
        _count(details, "cached_tokens") if isinstance(details, dict) else None,
        elapsed, float(cost) if cost is not None else None,
    )


class ModelSession:
    """Connector session holding usage and enforcing a shared model-call budget."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.usage: list[Usage] = []
        self.responses: list[dict[str, object]] = []
        self.diagnostics: list[dict[str, object]] = []

    def check_budget(self) -> None:
        if len(self.usage) >= self.budget.max_calls:
            raise BudgetExceeded(f"Model-call budget exhausted: {len(self.usage)}/{self.budget.max_calls}")
        if self.budget.cost_limit_usd is not None:
            if any(item.cost_usd is None for item in self.usage):
                raise BudgetExceeded("Cannot enforce a dollar budget with unpriced calls; configure pricing explicitly")
            total = sum(item.cost_usd or 0.0 for item in self.usage)
            if total >= self.budget.cost_limit_usd:
                raise BudgetExceeded(f"Dollar budget exhausted: {total:.6f}/{self.budget.cost_limit_usd:.6f}")

    def query(self, model: QueryModel, model_name: str, messages: list[Message], role: Role) -> Message:
        self.check_budget()
        started = time.monotonic()
        try:
            response = model.query(copy.deepcopy(messages))
            # Persist before parsing/selection so malformed outputs remain
            # available even when no decision can be produced.
            self.responses.append({"role": role, "model": model_name, "response": copy.deepcopy(response)})
            usage = response_usage(response, role, model_name, time.monotonic() - started)
        except Exception:
            # This logical query may already have been billed. Preserve an
            # explicitly unknown charge, then propagate the original failure.
            self.usage.append(Usage(role, model_name, None, None, None, time.monotonic() - started, None))
            raise
        self.usage.append(usage)
        if self.budget.cost_limit_usd is not None:
            if self.usage[-1].cost_usd is None:
                raise BudgetExceeded("Model call has no price; cannot continue with a dollar budget")
            total = sum(item.cost_usd or 0.0 for item in self.usage)
            if total > self.budget.cost_limit_usd:
                raise BudgetExceeded(f"Last call exceeded dollar budget: {total:.6f}/{self.budget.cost_limit_usd:.6f}")
        return response
