"""Run a recipe, summarize official evaluation outcomes, or construct verifier data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from minisweagent import Environment, Model
from minisweagent.config import builtin_config_dir
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

from swe_verifier.baselines.trajectory import Trajectory, oracle_at_k, select_trajectories
from swe_verifier.config import ModelSettings, Recipe, load_recipe, recipe_names
from swe_verifier.data.cli import main as data_main
from swe_verifier.evaluation import summarize_runs
from swe_verifier.inference.messages import policy_message
from swe_verifier.inference.models import LiteLLMTextModel
from swe_verifier.inference.native import NativeTextModel
from swe_verifier.inference.scoring import Scorer, SlateSelector
from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import Budget, BudgetExceeded, QueryModel, RuntimeConfig
from swe_verifier.integrations.mini import MiniVerifierAgent
from swe_verifier.prompts import load_prompt


def text_model(settings: ModelSettings, endpoint: str | None) -> QueryModel:
    options = dict(settings.kwargs)
    if settings.backend == "native":
        if endpoint is None or settings.revision is None:
            raise ValueError("Native verifier needs --gate-url/--selector-url and an immutable tokenizer revision")
        api_key = os.environ.get("SWE_VERIFIER_API_KEY")
        if api_key is not None:
            options["api_key"] = api_key
        return NativeTextModel(settings.name, endpoint, settings.name.removeprefix("openai/"),
                               settings.revision, options, settings.retries, settings.timeout_seconds)
    if endpoint is not None:
        options["api_base"] = endpoint
    options["timeout"] = settings.timeout_seconds
    return LiteLLMTextModel(settings.name, options, settings.retries)


def _policy(recipe: Recipe, endpoint: str | None) -> Model:
    options = dict(recipe.policy.kwargs)
    options["timeout"] = recipe.policy.timeout_seconds
    if endpoint is not None:
        options["api_base"] = endpoint
    return get_model(config={
        "model_class": "litellm", "model_name": recipe.policy.name, "model_kwargs": options,
        "observation_template": load_prompt("policy/observation.jinja"),
        "format_error_template": load_prompt("policy/format_error.jinja"),
    })


def _components(recipe: Recipe, args: argparse.Namespace) -> tuple[Scorer | None, Scorer | SlateSelector | None]:
    scorer: Scorer | None = None
    gate_endpoint = args.gate_url if recipe.strategy == "selective" else args.selector_url or args.gate_url
    if recipe.verifier is not None:
        model = text_model(recipe.verifier, gate_endpoint)
        scorer = Scorer(model, recipe.verifier.name, recipe.variant, recipe.scoring_draws, recipe.format_retries)
    gate = scorer if recipe.strategy == "selective" else None
    if recipe.selector_role == "none":
        return gate, None
    if recipe.selector_role == "policy":
        model = text_model(recipe.policy, args.selector_url or args.policy_url)
        return gate, SlateSelector(model, recipe.policy.name, recipe.format_retries)
    if args.selector_url and args.selector_url != gate_endpoint:
        if recipe.verifier is None:
            raise ValueError("Missing selector checkpoint")
        model = text_model(recipe.verifier, args.selector_url)
        scorer = Scorer(model, recipe.verifier.name, recipe.variant, recipe.scoring_draws, recipe.format_retries)
    return gate, scorer


def _task(args: argparse.Namespace) -> tuple[str, str, dict[str, object] | None]:
    if args.task_file is not None:
        if args.instance_id or args.instance_json:
            raise ValueError("Use --task-file or a SWE-bench instance, not both")
        return args.task_file.stem, args.task_file.read_text(encoding="utf-8"), None
    if args.instance_json is not None:
        instance = json.loads(args.instance_json.read_text(encoding="utf-8"))
    else:
        if args.instance_id is None:
            raise ValueError("Provide --instance-id, --instance-json, or --task-file")
        dataset = load_dataset(args.dataset, split=args.split, revision=args.dataset_revision)
        matches = [dict(row) for row in dataset if row["instance_id"] == args.instance_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one dataset instance {args.instance_id!r}, found {len(matches)}")
        instance = matches[0]
    if (not isinstance(instance, dict) or not isinstance(instance.get("problem_statement"), str)
            or not isinstance(instance.get("instance_id"), str)):
        raise ValueError("Instance JSON requires instance_id and problem_statement")
    return str(instance["instance_id"]), instance["problem_statement"], instance


def _environment(args: argparse.Namespace, instance: dict[str, object] | None) -> Environment:
    executable = args.docker_executable if args.environment_class == "docker" else args.singularity_executable
    config: dict[str, object] = {}
    if instance is not None:
        config = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())
        options = dict(config["environment"])
        image = args.image or get_swebench_docker_image_name(instance)
    elif args.image is None:
        raise ValueError("A custom task requires --image; run environments use isolated containers")
    else:
        options = {"cwd": args.cwd}
        image = args.image
    if args.environment_class == "singularity":
        if not Path(image).is_file() and "://" not in image:
            image = "docker://" + image
        # This is a Docker-only option in Mini's benchmark configuration.
        options.pop("interpreter", None)
        if args.singularity_exec_arg is not None:
            options["exec_args"] = args.singularity_exec_arg
    options.update(environment_class=args.environment_class, executable=executable, image=image)
    env = get_environment(options)
    startup = config.get("run", {}).get("env_startup_command")
    if startup and instance is not None:
        command = Template(startup, undefined=StrictUndefined).render(**instance)
        output = env.execute({"command": command})
        if output["returncode"] != 0:
            env.cleanup()
            raise RuntimeError(f"Error executing benchmark startup command: {output}")
    return env


def remaining_budget(recipe: Recipe, usage: list[dict[str, object]]) -> Budget:
    calls = recipe.max_calls - len(usage)
    if calls < 1:
        raise BudgetExceeded("Total run call budget exhausted before another rollout or selection")
    cost_limit = recipe.cost_limit_usd
    if cost_limit is not None:
        if any(item["cost_usd"] is None for item in usage):
            raise BudgetExceeded("Cannot enforce total cost budget because some calls have no price")
        cost_limit -= sum(float(item["cost_usd"]) for item in usage)
        if cost_limit <= 0:
            raise BudgetExceeded("Total run cost budget exhausted")
    return Budget(calls, cost_limit)


def run_recipe(args: argparse.Namespace) -> int:
    if args.list_recipes:
        print("\n".join(recipe_names()))
        return 0
    recipe = load_recipe(args.recipe)
    values = recipe.model_dump()
    if args.policy_model is not None:
        values["policy"] = {**values["policy"], "name": args.policy_model}
    for option in ("step_limit", "max_calls"):
        if getattr(args, option) is not None:
            values[option] = getattr(args, option)
    recipe = Recipe.model_validate(values)
    if args.dry_run:
        print(json.dumps(recipe.model_dump(), indent=2))
        return 0
    if args.output is None:
        raise ValueError("Live runs require an explicit --output directory")
    if args.output.exists():
        raise FileExistsError(f"Run output already exists: {args.output}; use a fresh directory")
    instance_id, task, instance = _task(args)
    gate, selector = _components(recipe, args)
    args.output.mkdir(parents=True)
    (args.output / "recipe.json").write_text(json.dumps(recipe.model_dump(), indent=2), encoding="utf-8")
    trajectories: list[Trajectory] = []
    saved: list[dict[str, object]] = []
    usage: list[dict[str, object]] = []
    selected: Trajectory | None = None
    status = "failed"
    count = recipe.rollout_count if recipe.strategy == "trajectory" else 1
    strategy = "none" if recipe.strategy == "trajectory" else recipe.strategy
    try:
        for index in range(count):
            budget = remaining_budget(recipe, usage)
            env = _environment(args, instance)
            agent: MiniVerifierAgent | None = None
            try:
                agent = MiniVerifierAgent(
                    _policy(recipe, args.policy_url), env,
                    verification=RuntimeConfig(strategy, recipe.candidate_count, recipe.threshold, recipe.failure_policy),
                    gate=gate, selector=selector, budget=budget,
                    system_template=load_prompt("policy/system.jinja"),
                    instance_template=load_prompt("policy/instance.jinja"),
                    step_limit=recipe.step_limit, cost_limit=recipe.policy_cost_limit_usd,
                    wall_time_limit_seconds=recipe.wall_time_limit_seconds,
                    output_path=args.output / f"rollout_{index}.traj.json",
                )
                result = agent.run(task, instance_id=instance_id)
                trace = json.dumps([policy_message(message) for message in agent.messages])
                trajectories.append(Trajectory(str(index), trace, str(result.get("submission", ""))))
            finally:
                if agent is not None:
                    saved.append(agent.serialize())
                    usage.extend(asdict(item) for item in agent.session.usage)
                cleanup = getattr(env, "cleanup", None)
                if callable(cleanup):
                    cleanup()
        if recipe.strategy == "trajectory":
            session = ModelSession(remaining_budget(recipe, usage))
            try:
                choice = select_trajectories(
                    task, tuple(trajectories), text_model(recipe.policy, args.selector_url or args.policy_url),
                    recipe.policy.name, recipe.trajectory_repetitions, session,
                )
                selected = choice.selected
                (args.output / "selection.json").write_text(json.dumps(asdict(choice), indent=2), encoding="utf-8")
            finally:
                usage.extend(asdict(item) for item in session.usage)
                (args.output / "selection-responses.json").write_text(
                    json.dumps(session.responses, indent=2), encoding="utf-8")
        else:
            selected = trajectories[0]
        status = "completed"
    finally:
        selected_record = saved[int(selected.id)] if selected is not None else saved[-1] if saved else None
        verification = cast(dict[str, object], selected_record["verification"]) if selected_record else {}
        decisions = cast(list[dict[str, object]], verification.get("decisions", []))
        error_type = sys.exc_info()[0]
        summary = {
            "schema_version": 1, "instance_id": instance_id, "recipe_name": recipe.name,
            "status": status, "error_type": error_type.__name__ if error_type is not None else None,
            "selected_rollout": selected.id if selected is not None else None,
            "selected_steps": len(decisions),
            "generated_steps": sum(len(record["verification"]["decisions"]) for record in saved),
            "verified_steps": sum(bool(decision["triggered"]) for decision in decisions), "usage": usage,
            "agent_exit_statuses": [record.get("info", {}).get("exit_status") for record in saved],
            "provenance_status": recipe.provenance_status,
            "dataset": args.dataset if args.instance_id is not None else None,
            "dataset_revision": args.dataset_revision if args.instance_id is not None else None,
            "input_source": "instance_json" if args.instance_json is not None else
                            "task_file" if args.task_file is not None else "dataset",
            "input_sha256": hashlib.sha256((args.instance_json or args.task_file).read_bytes()).hexdigest()
                            if args.instance_json is not None or args.task_file is not None else None,
            "environment_class": args.environment_class,
            "image_override": args.image,
        }
        prediction = {"instance_id": instance_id, "model_name_or_path": recipe.name,
                      "model_patch": selected.patch if status == "completed" and selected is not None else ""}
        (args.output / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
        (args.output / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"instance_id": instance_id, "output": str(args.output),
                      "selected_rollout": selected.id if selected is not None else None}))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    if args.oracle_outcomes is not None:
        groups = json.loads(args.oracle_outcomes.read_text(encoding="utf-8"))
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Oracle outcomes must map instance IDs to nonempty lists of booleans")
        values = {key: oracle_at_k(tuple(value)) for key, value in groups.items()}
        result = {"evaluation_only": True, "oracle_resolved": sum(values.values()), "instances": len(values)}
    else:
        if args.runs is None:
            raise ValueError("Provide --runs or --oracle-outcomes")
        result = summarize_runs(args.runs, args.report)
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "data":
        raise SystemExit(data_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one task with a packaged recipe")
    run.add_argument("--recipe", default="selective_two_stage")
    run.add_argument("--list-recipes", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--policy-model")
    run.add_argument("--policy-url")
    run.add_argument("--gate-url")
    run.add_argument("--selector-url")
    run.add_argument("--step-limit", type=int)
    run.add_argument("--max-calls", type=int)
    run.add_argument("--instance-id")
    run.add_argument("--instance-json", type=Path)
    run.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    run.add_argument("--dataset-revision", default="78f471bf655a3137b2e8a75af1501690ec009ec3")
    run.add_argument("--split", default="test")
    run.add_argument("--task-file", type=Path)
    run.add_argument("--image", help="Container image or local Singularity SIF; overrides the benchmark image")
    run.add_argument("--cwd", default="/testbed")
    run.add_argument("--docker-executable", default="docker")
    run.add_argument("--environment-class", choices=("docker", "singularity"), default="docker")
    run.add_argument("--singularity-executable", default="singularity")
    run.add_argument("--singularity-exec-arg", action="append",
                     help="Repeat to replace Mini's Singularity exec flags (for clusters without fakeroot mappings)")
    run.add_argument("--output", type=Path)
    run.set_defaults(handler=run_recipe)
    evaluation = commands.add_parser("evaluate", help="Import external SWE-bench results or summarize usage; does not run a grader")
    evaluation.add_argument("--runs", type=Path)
    evaluation.add_argument("--report", type=Path,
                            help="Aggregate JSON report from a separately configured SWE-bench harness")
    evaluation.add_argument("--oracle-outcomes", type=Path)
    evaluation.add_argument("--output", type=Path)
    evaluation.set_defaults(handler=evaluate)
    commands.add_parser("data", help="Resumable data construction; see data --help")
    args = parser.parse_args()
    raise SystemExit(args.handler(args))
