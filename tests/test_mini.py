from __future__ import annotations

import copy
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output

from swe_verifier.inference.scoring import Scorer
from swe_verifier.inference.types import Budget, RuntimeConfig
from swe_verifier.integrations.mini import MiniVerifierAgent


def test_mini_executes_only_selected_action_and_keeps_a_linear_history(tmp_path: Path) -> None:
    class HistoryModel(DeterministicModel):
        def query(self, messages: list[dict[str, object]]) -> dict[str, object]:
            self.last_input = copy.deepcopy(messages)
            return super().query(messages)

    policy = HistoryModel(outputs=[
        {**make_output("discarded first", [{"command": "printf wrong > wrong.txt"}], cost=0.01),
         "reasoning_content": "discarded reasoning"},
        {**make_output("selected", [{"command": "printf chosen > chosen.txt"}], cost=0.01),
         "reasoning_content": "selected reasoning"},
        make_output("discarded third", [{"command": "printf wrong > third.txt"}], cost=0.01),
        make_output("submit", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone'"}], cost=0.01),
    ])
    verifier = DeterministicModel(outputs=[make_output(f"SCORE: {score}", [], cost=0.01)
                                           for score in (0.1, 0.9, 0.2, 0.9)])
    scorer = Scorer(verifier, "verifier", "score_only", 1, 0)
    agent = MiniVerifierAgent(
        policy, LocalEnvironment(cwd=str(tmp_path)),
        verification=RuntimeConfig("selective", 3, 0.2, "raise"), gate=scorer, selector=scorer,
        budget=Budget(20, 5.0), system_template="Solve the task", instance_template="{{ task }}",
        step_limit=250, cost_limit=5.0, output_path=tmp_path / "trajectory.json",
    )
    result = agent.run("Write the chosen file")
    assert result["exit_status"] == "Submitted"
    assert (tmp_path / "chosen.txt").read_text() == "chosen"
    assert not (tmp_path / "wrong.txt").exists()
    assert not (tmp_path / "third.txt").exists()
    assert [message["content"] for message in agent.messages if message["role"] == "assistant"] == ["selected", "submit"]
    serialized = agent.serialize()
    assert len(serialized["verification"]["usage"]) == 8
    assert len(serialized["verification"]["decisions"]) == 2
    assert "SCORE" not in str(serialized["messages"])
    retained = [message for message in policy.last_input if message["role"] == "assistant"]
    assert retained[0]["reasoning_content"] == "selected reasoning"
    assert "discarded reasoning" not in str(policy.last_input)
    assert "SCORE" not in str(policy.last_input)
    assert all("extra" not in message for message in policy.last_input)
    assert len(serialized["verification"]["responses"]) == 8


def test_policy_format_error_is_charged_once_and_counts_against_shared_budget(tmp_path: Path) -> None:
    class FormatOnceModel(DeterministicModel):
        def __init__(self):
            super().__init__(outputs=[make_output(
                "submit", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone'"}], cost=0.05)])
            self.failed = False

        def query(self, messages: list[dict[str, object]]) -> dict[str, object]:
            if not self.failed:
                self.failed = True
                raise FormatError({"role": "user", "content": "Use one bash action", "extra": {"cost": 0.07}})
            return super().query(messages)

    policy = FormatOnceModel()
    agent = MiniVerifierAgent(
        policy, LocalEnvironment(cwd=str(tmp_path)), verification=RuntimeConfig("none", 1, 0.2, "raise"),
        gate=None, selector=None, budget=Budget(2, 1.0), system_template="Solve", instance_template="{{ task }}",
        step_limit=250, cost_limit=1.0, output_path=None,
    )
    result = agent.run("Submit")
    assert result["exit_status"] == "Submitted"
    assert len(agent.session.usage) == 2
    assert round(agent.cost, 2) == 0.12
    assert [item.cost_usd for item in agent.session.usage] == [0.07, 0.05]
