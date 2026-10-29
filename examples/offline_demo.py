"""Exercise every release strategy with deterministic outputs and real local execution.

No model API, container engine, or checkpoint download is needed. Only the chosen
command executes, inside a newly created temporary directory. The outputs show
framework behavior, not benchmark performance or model quality.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from swe_verifier.baselines.trajectory import Trajectory, oracle_at_k, select_trajectories
from swe_verifier.inference.engine import choose_action
from swe_verifier.inference.scoring import Scorer, SlateSelector
from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import Budget, RuntimeConfig, Variant
from swe_verifier.integrations.mini import MiniVerifierAgent


def _policy() -> DeterministicModel:
    return DeterministicModel(outputs=[
        make_output(name, [{"command": f"printf {name}"}], cost=0.01)
        for name in ("initial", "selected", "alternative")
    ])


def _verifier(variant: Variant, scores: tuple[float, ...]) -> DeterministicModel:
    outputs = []
    for score in scores:
        if variant == "two_stage":
            outputs.append(make_output(
                "<checklist>\nREASONING: Evaluate this proposed command.\n"
                "1. The command makes progress [weight: must]\n</checklist>", [], cost=0.01))
            text = (f"<checklist_scores>\n- Item 1: {score:.2f}\n  REASONING: The command helps.\n"
                    f"</checklist_scores>\nREASONING: It advances the task.\nSCORE: {score:.2f}")
        else:
            text = f"REASONING: Deterministic demonstration.\nSCORE: {score:.2f}"
        outputs.append(make_output(text, [], cost=0.01))
    return DeterministicModel(outputs=outputs)


def demonstrate(output: Path) -> dict[str, object]:
    """Run all strategy paths and write a JSON receipt to a new file."""
    receipts: dict[str, object] = {}
    for strategy in ("none", "all_step", "selective"):
        variants: tuple[Variant, ...] = (("untrained", "score_only", "joint_rubric", "two_stage")
                                        if strategy == "selective" else ("untrained",))
        for variant in variants:
            scorer = Scorer(_verifier(variant, (0.1, 0.9, 0.2)), "demonstration", variant, 1, 0)
            decision = choose_action(
                "Choose a useful action", [], _policy(), "scripted-policy",
                scorer if strategy == "selective" else None,
                scorer if strategy != "none" else None,
                RuntimeConfig(strategy, 3, 0.2, "raise"), ModelSession(Budget(30, 2.0)),
            )
            receipts[f"{strategy}_{variant}"] = asdict(decision)
    gate = Scorer(_verifier("score_only", (0.1,)), "small-gate", "score_only", 1, 0)
    selector = SlateSelector(DeterministicModel(outputs=[make_output("FINAL: 2", [], cost=0.01)]),
                             "policy-selector", 0)
    receipts["gate_only"] = asdict(choose_action(
        "Choose a useful action", [], _policy(), "scripted-policy", gate, selector,
        RuntimeConfig("selective", 3, 0.2, "raise"), ModelSession(Budget(30, 2.0)),
    ))
    judge = DeterministicModel(outputs=[make_output("<score_A>A</score_A><score_B>T</score_B>", [], cost=0.01)] * 9)
    receipts["trajectory"] = asdict(select_trajectories(
        "Choose a complete solution", tuple(Trajectory(str(i), f"trace {i}", f"patch {i}") for i in range(3)),
        judge, "scripted-policy", 1, ModelSession(Budget(30, 2.0)),
    ))
    receipts["oracle_at_3"] = oracle_at_k((False, True, False))
    with tempfile.TemporaryDirectory(prefix="swe-verifier-demo-") as directory:
        cwd = Path(directory)
        policy = DeterministicModel(outputs=[
            make_output("unselected", [{"command": "printf wrong > wrong.txt"}], cost=0.01),
            make_output("selected", [{"command": "printf correct > chosen.txt"}], cost=0.01),
            make_output("alternative", [{"command": "printf wrong > alternative.txt"}], cost=0.01),
            make_output("submit", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone'"}], cost=0.01),
        ])
        scorer = Scorer(_verifier("score_only", (0.1, 0.9, 0.2, 0.9)), "demo-verifier", "score_only", 1, 0)
        agent = MiniVerifierAgent(
            policy, LocalEnvironment(cwd=directory), verification=RuntimeConfig("selective", 3, 0.2, "raise"),
            gate=scorer, selector=scorer, budget=Budget(30, 2.0),
            system_template="Run the requested command", instance_template="{{ task }}", step_limit=250,
            cost_limit=2.0, output_path=None,
        )
        result = agent.run("Write chosen.txt")
        if not (cwd / "chosen.txt").is_file() or (cwd / "wrong.txt").exists() or (cwd / "alternative.txt").exists():
            raise RuntimeError("Mini demonstration executed an unexpected action")
        receipts["mini_execution"] = {"result": result, "chosen_text": (cwd / "chosen.txt").read_text(),
                                       "verification": agent.serialize()["verification"]}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipts, handle, indent=2)
        handle.write("\n")
    return {"strategies_exercised": list(receipts), "output": str(output), "model_api_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(demonstrate(args.output), indent=2))


if __name__ == "__main__":
    main()
