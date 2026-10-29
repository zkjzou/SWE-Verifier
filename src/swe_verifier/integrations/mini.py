"""Mini-swe-agent adapter; verification stays outside the executed policy history."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict

from minisweagent import Environment, Model
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded, TimeExceeded

from swe_verifier.inference.engine import choose_action
from swe_verifier.inference.scoring import Scorer, SlateSelector
from swe_verifier.inference.session import ModelSession, response_usage
from swe_verifier.inference.types import Budget, BudgetExceeded, Decision, Message, RuntimeConfig


class MiniVerifierAgent(DefaultAgent):
    """Override only action-query orchestration, retaining Mini execution and save formats.

    agent.step_limit counts executed decision steps; max_calls counts all policy,
    gate, rubric, selector, and corrective-generation calls in the model session.
    """

    def __init__(self, model: Model, env: Environment, *, verification: RuntimeConfig,
                 gate: Scorer | None, selector: Scorer | SlateSelector | None, budget: Budget, **kwargs: object):
        super().__init__(model, env, **kwargs)
        self.verification, self.gate, self.selector = verification, gate, selector
        self.session = ModelSession(budget)
        self.decisions: list[Decision] = []

    def run(self, task: str, **kwargs: object) -> dict[str, object]:
        self.session = ModelSession(self.session.budget)
        self.decisions = []
        self.cost, self.n_calls, self.n_consecutive_format_errors = 0.0, 0, 0
        self._start_time = time.time()
        return super().run(task, **kwargs)

    def query(self) -> Message:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({"role": "exit", "content": "LimitsExceeded",
                                  "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        if 0 < self.config.wall_time_limit_seconds <= time.time() - self._start_time:
            raise TimeExceeded({"role": "exit", "content": "TimeExceeded",
                                "extra": {"exit_status": "TimeExceeded", "submission": ""}})
        self.n_calls += 1
        model_name = str(self.model.get_template_vars().get("model_name", type(self.model).__name__))
        try:
            decision = choose_action(
                str(self.extra_template_vars.get("task", "")), self.messages, self.model, model_name,
                self.gate, self.selector, self.verification, self.session,
            )
        except FormatError as exc:
            # Mini's run loop charges this failed policy parse itself. Preserve
            # the same call in our shared budget and serialized role accounting.
            if exc.messages:
                failed_usage = response_usage(exc.messages[0], "policy", model_name, 0.0)
                self.session.usage[-1] = failed_usage
                # The base run loop adds the failed response cost immediately.
                self.cost = sum(item.cost_usd or 0.0 for item in self.session.usage) - (failed_usage.cost_usd or 0.0)
            raise
        except BudgetExceeded as exc:
            self.cost = sum(item.cost_usd or 0.0 for item in self.session.usage)
            raise LimitsExceeded({"role": "exit", "content": str(exc),
                                  "extra": {"exit_status": "LimitsExceeded", "submission": ""}}) from exc
        self.decisions.append(decision)
        self.cost = sum(item.cost_usd or 0.0 for item in self.session.usage)
        message = copy.deepcopy(decision.selected.message)
        self.add_messages(message)
        return message

    def serialize(self, *extra_dicts: dict[str, object]) -> dict[str, object]:
        result = super().serialize(*extra_dicts)
        result["info"]["model_stats"]["instance_cost"] = sum(item.cost_usd or 0.0 for item in self.session.usage)
        result["verification"] = {
            "config": asdict(self.verification), "budget": asdict(self.session.budget),
            "decisions": [asdict(decision) for decision in self.decisions],
            "usage": [asdict(usage) for usage in self.session.usage],
            "responses": copy.deepcopy(self.session.responses),
            "diagnostics": copy.deepcopy(self.session.diagnostics),
            "cost_complete": all(usage.cost_usd is not None for usage in self.session.usage),
        }
        return result
