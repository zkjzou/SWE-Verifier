from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from swe_verifier import cli
from swe_verifier.config import ModelSettings, Recipe, load_recipe, recipe_names
from swe_verifier.inference.types import Message


def _invoke(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["MSWEA_SILENT_STARTUP"] = "1"
    environment["MSWEA_GLOBAL_CONFIG_DIR"] = str(cwd / "mini-config")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run([sys.executable, "-m", "swe_verifier", *arguments], cwd=cwd, env=environment,
                          capture_output=True, text=True, check=False, timeout=60)


def test_packaged_recipes_validate_and_cli_resolves_outside_checkout(tmp_path: Path) -> None:
    names = recipe_names()
    assert {"none", "all_step_base", "all_step_self", "trajectory", "gate_only", "selective_score_only",
            "selective_joint_rubric", "selective_two_stage", "selective_untrained"} <= set(names)
    for name in names:
        recipe = load_recipe(name)
        assert recipe.scoring_draws == 3
        omitted = recipe.model_dump()
        omitted.pop("scoring_draws")
        assert Recipe.model_validate(omitted).scoring_draws == 3
        assert recipe.step_limit == 250
        if recipe.verifier is not None:
            assert len(recipe.verifier.revision) == 40
    process = _invoke(["run", "--recipe", "gate_only", "--dry-run"], tmp_path)
    assert process.returncode == 0, process.stderr
    receipt = json.loads(process.stdout)
    assert receipt["selector_role"] == "policy"
    assert receipt["strategy"] == "selective"
    assert receipt["scoring_draws"] == 3
    assert not list(tmp_path.glob("*.traj.json"))


def test_cli_offline_evaluation_preserves_unknown_costs_and_joins_report(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    for name, verified in (("a", 1), ("b", 0)):
        directory = runs / name
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps({
            "instance_id": name, "recipe_name": "selective_score_only", "status": "completed", "selected_steps": 2,
            "generated_steps": 2, "verified_steps": verified,
            "usage": [{"role": "gate", "cost_usd": None, "latency_seconds": 0.5}],
        }))
    report = tmp_path / "official.json"
    report.write_text(json.dumps({"submitted_ids": ["a", "b"], "completed_ids": ["a", "b"], "resolved_ids": ["b"], "unresolved_ids": ["a"]}))
    process = _invoke(["evaluate", "--runs", str(runs), "--report", str(report)], tmp_path)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["resolve_rate"] == 0.5
    assert result["verify_rate"] == 0.25
    assert result["known_cost_usd"] == 0.0
    assert not result["cost_complete"]


def test_oracle_is_only_available_in_offline_evaluation(tmp_path: Path) -> None:
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps({"a": [False, True, False], "b": [False, False, False]}))
    process = _invoke(["evaluate", "--oracle-outcomes", str(outcomes)], tmp_path)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"evaluation_only": True, "oracle_resolved": 1, "instances": 2}
    outcomes.write_text(json.dumps({"a": ["true"]}))
    invalid = _invoke(["evaluate", "--oracle-outcomes", str(outcomes)], tmp_path)
    assert invalid.returncode != 0
    assert "boolean" in invalid.stderr


def test_offline_demo_executes_every_strategy_with_no_remote_model(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "offline_demo.py"
    output = tmp_path / "demo.json"
    environment = dict(os.environ)
    environment["MSWEA_SILENT_STARTUP"] = "1"
    environment["MSWEA_GLOBAL_CONFIG_DIR"] = str(tmp_path / "mini-config")
    environment["PYTHONPATH"] = str(example.parents[1] / "src")
    process = subprocess.run([sys.executable, str(example), "--output", str(output)], cwd=tmp_path, env=environment,
                             capture_output=True, text=True, check=False, timeout=60)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["model_api_calls"] == 0
    receipt = json.loads(output.read_text())
    assert receipt["mini_execution"]["chosen_text"] == "correct"
    assert receipt["selective_two_stage"]["selected"]["id"] == "1"
    assert receipt["gate_only"]["selected"]["id"] == "1"
    assert receipt["oracle_at_3"] is True


def test_trajectory_cli_judges_visible_history_and_preserves_raw_logs(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "fixture.txt").write_text("visible" + "RAW_TOOL_SENTINEL" * 1000)
    task = tmp_path / "task.txt"
    task.write_text("Inspect the fixture and submit a patch")
    output = tmp_path / "run"

    class Judge(DeterministicModel):
        def query(self, messages: list[Message]) -> Message:
            prompt = str(messages)
            assert "RAW_TOOL_SENTINEL" not in prompt
            assert "reasoning kept" in prompt
            assert "visible" in prompt
            return super().query(messages)

    judge = Judge(outputs=[make_output("<score_A>A</score_A><score_B>T</score_B>", [], cost=0.0)] * 36)

    def policy(recipe: Recipe, endpoint: str | None) -> DeterministicModel:
        return DeterministicModel(observation_template="{{ output.output[:7] }}", outputs=[
            {**make_output("inspect", [{"command": "cat fixture.txt"}], cost=0.0),
             "reasoning_content": "reasoning kept"},
            make_output("submit", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch'"}], cost=0.0),
        ])

    def environment(args: argparse.Namespace, instance: dict[str, object] | None) -> LocalEnvironment:
        return LocalEnvironment(cwd=str(tmp_path))

    def selector(settings: ModelSettings, endpoint: str | None) -> Judge:
        return judge

    monkeypatch.setattr(cli, "_policy", policy)
    monkeypatch.setattr(cli, "_environment", environment)
    monkeypatch.setattr(cli, "text_model", selector)
    monkeypatch.setattr(sys, "argv", ["swe-verifier", "run", "--recipe", "trajectory",
                                    "--task-file", str(task), "--output", str(output)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert judge.current_index == 35
    assert "RAW_TOOL_SENTINEL" in (output / "rollout_0.traj.json").read_text()
    assert "RAW_TOOL_SENTINEL" not in (output / "selection.json").read_text()
    assert len(json.loads((output / "selection-responses.json").read_text())) == 36
    assert json.loads((output / "predictions.jsonl").read_text())["model_patch"] == "patch"
