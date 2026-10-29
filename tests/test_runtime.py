from __future__ import annotations

import copy

import pytest

from swe_verifier.baselines.trajectory import Trajectory, oracle_at_k, select_trajectories
from swe_verifier.inference.engine import choose_action
from swe_verifier.inference.messages import observed_history
from swe_verifier.inference.scoring import Scorer, SlateSelector, terminal_score
from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import Budget, BudgetExceeded, Candidate, InvalidOutput, Message, RuntimeConfig


class RecordedModel:
    """A scripted model connector recording the exact public message boundary."""

    def __init__(self, outputs: list[Message]):
        self.outputs = outputs
        self.inputs: list[list[Message]] = []

    def query(self, messages: list[Message]) -> Message:
        self.inputs.append(copy.deepcopy(messages))
        return copy.deepcopy(self.outputs[len(self.inputs) - 1])


def action(text: str) -> Message:
    return {"role": "assistant", "content": f"```bash\n{text}\n```", "extra": {
        "actions": [{"command": text}], "cost": 0.1, "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }}


def completion(text: str) -> Message:
    return {"role": "assistant", "content": text, "extra": {"cost": 0.01}}


def test_threshold_equality_is_pass_through_and_history_is_unchanged() -> None:
    history = [{"role": "user", "content": "Fix this"}]
    policy = RecordedModel([action("echo initial")])
    verifier = RecordedModel([completion("REASONING: fine\nSCORE: 0.20")])
    scorer = Scorer(verifier, "gate", "score_only", 1, 0)
    result = choose_action("Fix this", history, policy, "policy", scorer, scorer,
                           RuntimeConfig("selective", 3, 0.2, "raise"), ModelSession(Budget(10, 3.0)))
    assert not result.triggered
    assert result.selected.id == "0"
    assert len(policy.inputs) == 1
    assert history == [{"role": "user", "content": "Fix this"}]


def test_selective_reuses_gate_score_and_samples_from_identical_state() -> None:
    history = [{"role": "user", "content": "Fix this"}]
    policy = RecordedModel([action("echo first"), action("echo best"), action("echo third")])
    verifier = RecordedModel([completion(f"SCORE: {score}") for score in (0.1, 0.9, 0.3)])
    scorer = Scorer(verifier, "small", "score_only", 1, 0)
    result = choose_action("Fix this", history, policy, "policy", scorer, scorer,
                           RuntimeConfig("selective", 3, 0.2, "raise"), ModelSession(Budget(10, 3.0)))
    assert result.selected.id == "1"
    assert len(verifier.inputs) == 3
    assert all(messages == history for messages in policy.inputs)
    assert [item.role for item in result.usage] == ["policy", "gate", "policy", "policy", "selector", "selector"]


def test_gate_only_invokes_policy_selector_and_does_not_rank_with_gate() -> None:
    policy = RecordedModel([action("echo one"), action("echo two"), action("echo three")])
    gate_model = RecordedModel([completion("SCORE: 0.01")])
    selector_model = RecordedModel([completion("REASONING: best\nFINAL: 2")])
    gate = Scorer(gate_model, "gate", "score_only", 1, 0)
    selector = SlateSelector(selector_model, "policy-selector", 0)
    result = choose_action("task", [], policy, "policy", gate, selector,
                           RuntimeConfig("selective", 3, 0.2, "raise"), ModelSession(Budget(10, 3.0)))
    assert result.selected.id == "1"
    assert len(gate_model.inputs) == len(selector_model.inputs) == 1
    assert "echo two" in str(selector_model.inputs[0])
    assert result.scores == ()


def test_two_stage_rubrics_are_candidate_local_and_observations_are_absent() -> None:
    def rubric(label: str) -> str:
        return f"<checklist>\nREASONING: Check {label}.\n1. Validate {label} [weight: must]\n</checklist>"
    def score(value: str) -> str:
        return (f"<checklist_scores>\n- Item 1: {value}\n  REASONING: The command helps.\n</checklist_scores>\n"
                f"REASONING: This command advances the task.\nSCORE: {value}")
    policy = RecordedModel([action("echo original"), action("echo alternative")])
    verifier = RecordedModel([completion(rubric("original")), completion(score("0.10")),
                              completion(rubric("alternative")), completion(score("0.90"))])
    scorer = Scorer(verifier, "shared-checkpoint", "two_stage", 1, 0)
    result = choose_action("task", [], policy, "policy", scorer, scorer,
                           RuntimeConfig("selective", 2, 0.2, "raise"), ModelSession(Budget(20, 3.0)))
    assert result.selected.id == "1"
    assert len(verifier.inputs) == 4
    assert "Validate original" in str(verifier.inputs[1][-1])
    assert "Validate alternative" in str(verifier.inputs[3][-1])
    assert "Validate original" not in str(verifier.inputs[3])
    assert all(message["role"] != "tool" for messages in verifier.inputs for message in messages)


def test_draw_averaging_and_strict_invalid_output() -> None:
    model = RecordedModel([completion(f"SCORE: {value}") for value in (0.1, 0.4, 0.7)])
    result = Scorer(model, "score", "score_only", 3, 0).score(
        "task", [], Candidate("a", action("pwd")), "gate", ModelSession(Budget(3, None)))
    assert result.value == pytest.approx(0.4)
    for value in ("SCORE: nan", "SCORE: 1.1", "SCORE: 0.2\nSCORE: 0.3", "reasoning 0.9"):
        with pytest.raises(InvalidOutput):
            terminal_score(value)


def test_explicit_fallback_and_budget_failure_are_observable() -> None:
    policy = RecordedModel([action("pwd")])
    scorer = Scorer(RecordedModel([completion("unparseable")]), "score", "score_only", 1, 0)
    result = choose_action("task", [], policy, "policy", scorer, scorer,
                           RuntimeConfig("selective", 3, 0.2, "first_valid"), ModelSession(Budget(5, None)))
    assert result.fallback_reason
    assert result.selected.id == "0"
    policy = RecordedModel([action("pwd")])
    scorer = Scorer(RecordedModel([completion("SCORE: 0.1")]), "score", "score_only", 1, 0)
    session = ModelSession(Budget(2, None))
    with pytest.raises(BudgetExceeded):
        choose_action("task", [], policy, "policy", scorer, scorer,
                      RuntimeConfig("selective", 3, 0.2, "raise"), session)
    assert len(session.usage) == 2


def test_all_step_filters_invalid_alternatives_before_selection() -> None:
    invalid = {"role": "assistant", "content": "no action", "extra": {"cost": 0.0, "actions": []}}
    policy = RecordedModel([action("pwd"), invalid, action("ls")])
    selector = Scorer(RecordedModel([completion("SCORE: 0.4"), completion("SCORE: 0.8")]), "score", "untrained", 1, 0)
    result = choose_action("task", [], policy, "policy", None, selector,
                           RuntimeConfig("all_step", 3, 0.2, "raise"), ModelSession(Budget(8, None)))
    assert result.selected.id == "2"
    assert result.fallback_reason is not None


def test_trajectory_tournament_and_offline_oracle() -> None:
    trajectories = tuple(Trajectory(str(i), f"trace {i}", f"patch {i}") for i in range(3))
    model = RecordedModel([completion("<score_A>A</score_A><score_B>T</score_B>")] * 9)
    result = select_trajectories("task", trajectories, model, "judge", 1, ModelSession(Budget(9, None)))
    assert result.selected.id == "0"
    assert result.wins == (2.0, 1.0, 0.0)
    assert len(result.comparisons) == 9
    assert all(item.scoring_mode == "parsed_letters" for item in result.comparisons)
    assert oracle_at_k((False, True, False))
    assert not oracle_at_k((False, False, False))


def test_two_stage_corrective_retry_increases_only_retry_output_budget() -> None:
    class BudgetedModel(RecordedModel):
        def __init__(self, outputs: list[Message]):
            super().__init__(outputs)
            self.retry_limits: list[int] = []

        def query_with_output_limit(self, messages: list[Message], max_tokens: int) -> Message:
            self.retry_limits.append(max_tokens)
            return self.query(messages)

    rubric = "<checklist>\nREASONING: Check progress.\n1. Advances the task [weight: must]\n</checklist>"
    valid = ("<checklist_scores>\n- Item 1: 0.80\n  REASONING: Helpful.\n</checklist_scores>\n"
             "REASONING: The command advances the task.\nSCORE: 0.80")
    model = BudgetedModel([completion("bad rubric"), completion(rubric), completion(valid)])
    session = ModelSession(Budget(10, None))
    result = Scorer(model, "shared", "two_stage", 1, 1).score("task", [], Candidate("a", action("pwd")), "gate", session)
    assert result.value == 0.8
    assert model.retry_limits == [4096]
    assert len(session.usage) == 3
    assert "bad rubric" in str(model.inputs[1])
    assert "bad rubric" not in str(model.inputs[2])


def test_all_step_fallback_still_counts_as_verified_step() -> None:
    policy = RecordedModel([action("pwd"), action("ls")])
    selector = Scorer(RecordedModel([completion("invalid score"), completion("invalid score")]), "score", "score_only", 1, 0)
    result = choose_action("task", [], policy, "policy", None, selector,
                           RuntimeConfig("all_step", 2, 0.2, "first_valid"), ModelSession(Budget(5, None)))
    assert result.triggered
    assert result.fallback_reason
    assert result.selected.id == "0"


def test_slate_selector_can_correct_a_malformed_index() -> None:
    model = RecordedModel([completion("FINAL: 9"), completion("FINAL: 2")])
    session = ModelSession(Budget(3, None))
    selected, outputs = SlateSelector(model, "policy", 1).select(
        "task", [], (Candidate("a", action("pwd")), Candidate("b", action("ls"))), session)
    assert selected.id == "b"
    assert len(outputs) == len(session.usage) == 2
    assert "FINAL: 9" in str(model.inputs[-1])


def test_native_format_correction_is_not_an_observed_tool_result() -> None:
    rejected = {"role": "assistant", "content": "invalid call", "tool_calls": [{
        "id": "bad-call", "type": "function", "function": {"name": "unknown", "arguments": "{}"},
    }]}
    history = [rejected, {"role": "user", "content": "Use the bash tool"}]
    assert observed_history(history) == []
    text_history = [action("pwd"), {"role": "user", "content": "/testbed"}]
    observed = observed_history(text_history)
    assert [message["role"] for message in observed] == ["assistant", "tool"]
    assert observed[1]["content"] == "/testbed"


def test_scalar_prompt_layout_matches_source_and_contains_unexecuted_command() -> None:
    model = RecordedModel([completion("SCORE: 0.7")])
    scorer = Scorer(model, "small", "score_only", 1, 0)
    scorer.score("Fix task", [], Candidate("x", action("ls -la /testbed")), "gate", ModelSession(Budget(2, None)))
    assert [message["role"] for message in model.inputs[0]] == ["system", "user"]
    assert "Fix task" in str(model.inputs[0][-1]["content"])
    assert "ls -la /testbed" in str(model.inputs[0][-1]["content"])


def test_failed_model_call_records_unknown_cost_without_hiding_error() -> None:
    class FailingModel:
        def query(self, messages: list[Message]) -> Message:
            raise ConnectionError("deliberate connection failure")

    session = ModelSession(Budget(2, None))
    with pytest.raises(ConnectionError, match="deliberate"):
        session.query(FailingModel(), "remote", [], "selector")
    assert len(session.usage) == 1
    assert session.usage[0].role == "selector"
    assert session.usage[0].cost_usd is None


def test_research_partial_draws_and_missing_gate_do_not_resample() -> None:
    session = ModelSession(Budget(10, None))
    model = RecordedModel([completion(text) for text in ("missing", "reason SCORE: 0.2 trailing", "SCORE: 0.8")])
    result = Scorer(model, "score", "untrained", 3, 0).score(
        "task", [], Candidate("a", action("pwd")), "gate", session)
    assert result.value == 0.5
    assert result.samples == (None, 0.2, 0.8)
    assert len(session.usage) == 3
    assert session.diagnostics[0]["event"] == "missing_scalar_draw"
    policy = RecordedModel([action("pwd")])
    gate = Scorer(RecordedModel([completion("missing")]), "gate", "score_only", 1, 0)
    result = choose_action("task", [], policy, "policy", gate, gate,
                           RuntimeConfig("selective", 3, 0.2, "first_valid"), session)
    assert not result.triggered
    assert len(policy.inputs) == 1
    assert session.diagnostics[-1]["event"] == "missing_gate_score"


def test_research_slate_index_and_logged_fallback() -> None:
    for text, expected in (("CHOICE: Candidate 2\nExplanation", "1"), ("FINAL: 0.90", "0")):
        policy = RecordedModel([action("pwd"), action("ls")])
        selector = SlateSelector(RecordedModel([completion(text)]), "policy", 0)
        session = ModelSession(Budget(4, None))
        result = choose_action("task", [], policy, "policy", None, selector,
                               RuntimeConfig("all_step", 2, 0.2, "first_valid"), session)
        assert result.selected.id == expected
        if text == "FINAL: 0.90":
            assert result.fallback_reason
            assert session.diagnostics[-1]["event"] == "first_valid_fallback"
