"""Stage-oriented CLI; call main(argv) from the package's `data` command."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import yaml
from minisweagent.models import get_model

from swe_verifier.data.adapters import CombinedSampler, DockerReplay, ModelSampler, ModelTeacher
from swe_verifier.data.importers import collect_mini_index, source_fingerprints
from swe_verifier.data.pipeline import (
    checked_provenance,
    collect_successful,
    deduplicate_observations,
    export_dataset,
    filter_actions,
    label_candidates,
    read_jsonl,
    replay_candidates,
    required_string,
    run_stage,
    sample_candidates,
    validate_slate,
)
from swe_verifier.inference.models import LiteLLMTextModel
from swe_verifier.prompts import load_prompt


def _config(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("Data config must be a YAML mapping")
    serialized = json.dumps(loaded).lower()
    if any(secret in serialized for secret in ('"api_key"', '"authorization"', '"access_token"')):
        raise ValueError("Keep credentials in environment variables, not persisted data configs")
    return loaded


def _model(spec: dict[str, object]) -> LiteLLMTextModel:
    kwargs = spec.get("model_kwargs")
    retries = spec.get("retries")
    if not isinstance(kwargs, dict) or type(retries) is not int or retries < 1:
        raise ValueError("Each model spec requires model_kwargs mapping and retries >= 1")
    return LiteLLMTextModel(required_string(spec, "model_name"), kwargs, retries)


def _sampler(config: dict[str, object]) -> CombinedSampler:
    specs = config.get("samplers")
    if not isinstance(specs, list) or not specs:
        raise ValueError("Config requires a nonempty samplers list")
    samplers: list[ModelSampler] = []
    identities: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each sampler spec must be a mapping")
        provenance = checked_provenance(spec.get("provenance"))
        identity = json.dumps(provenance, sort_keys=True)
        if identity in identities:
            raise ValueError("Sampler provenance must distinguish each configured model")
        identities.add(identity)
        samples = spec.get("samples")
        if type(samples) is not int:
            raise ValueError("Each sampler requires an explicit integer samples count")
        kwargs = spec.get("model_kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError("Sampler model_kwargs must be a mapping")
        policy = get_model(
            config={
                "model_name": required_string(spec, "model_name"),
                "model_class": required_string(spec, "model_class"),
                "model_kwargs": kwargs,
            }
        )
        samplers.append(ModelSampler(policy, provenance, samples))
    return CombinedSampler(samplers)


def _teacher(config: dict[str, object]) -> ModelTeacher:
    spec = config.get("teacher")
    if not isinstance(spec, dict):
        raise ValueError("Config requires a teacher mapping")
    return ModelTeacher(_model(spec), checked_provenance(spec.get("provenance")))


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="swe-verifier data", description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("collect", "collect-mini", "sample", "filter", "replay", "deduplicate", "label", "export", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
        if name in {"sample", "label"}:
            command.add_argument("--config", type=Path, required=True)
        if name == "filter":
            command.add_argument("--threshold", type=float, default=0.8)
        if name == "replay":
            command.add_argument("--docker", default="docker")
            command.add_argument("--timeout", type=int, default=120)
        if name == "export":
            command.add_argument("--seed", required=True)
            command.add_argument("--validation-fraction", type=float, required=True)
    args = parser.parse_args(list(argv))
    stage = cast(str, args.stage)
    source = cast(Path, args.input)
    if stage == "validate":
        seen: set[str] = set()
        count = 0
        for row in read_jsonl(source):
            slate = validate_slate(row)
            if slate["slate_id"] in seen:
                raise ValueError(f"Duplicate slate_id: {slate['slate_id']}")
            seen.add(slate["slate_id"])
            count += 1
        print(json.dumps({"slates": count, "valid": True}))
        return 0
    output = cast(Path, args.output)
    if stage == "export":
        counts = export_dataset(source, output, args.seed, args.validation_fraction)
        print(json.dumps({"output": str(output), "actions": counts}))
        return 0
    settings: dict[str, object] = {
        "adapters_sha256": hashlib.sha256(Path(__file__).with_name("adapters.py").read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if stage == "collect":
        transform = collect_successful
    elif stage == "collect-mini":
        settings["trajectory_files"] = source_fingerprints(source)
        transform = functools.partial(collect_mini_index, base_dir=source.parent)
    elif stage == "filter":
        settings["threshold"] = args.threshold
        transform = functools.partial(filter_actions, threshold=args.threshold)
    elif stage == "deduplicate":
        transform = deduplicate_observations
    elif stage == "sample":
        config = _config(args.config)
        settings["config"] = config
        transform = functools.partial(sample_candidates, sampler=_sampler(config))
    elif stage == "replay":
        images: dict[str, str] = {}
        for row in read_jsonl(source):
            slate = validate_slate(row)
            image = slate["replay"].get("image")
            if not image:
                raise ValueError("Replay requires an image in each slate")
            if image not in images:
                result = subprocess.run(
                    [args.docker, "image", "inspect", "--format", "{{.Id}}", image],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=args.timeout,
                )
                images[image] = result.stdout.strip()
        settings.update({"docker": args.docker, "timeout": args.timeout, "image_revisions": images})
        transform = functools.partial(replay_candidates, replayer=DockerReplay(args.docker, args.timeout, images))
    elif stage == "label":
        config = _config(args.config)
        settings["config"] = config
        settings["teacher_prompts"] = {
            name: hashlib.sha256(load_prompt(name).encode()).hexdigest()
            for name in ("teacher/system.txt", "teacher/annotate.jinja")
        }
        transform = functools.partial(label_candidates, teacher=_teacher(config))
    else:
        raise ValueError(f"Unknown data stage: {stage}")
    result = run_stage(source, output, stage, settings, transform)
    print(json.dumps({"stage": stage, "output": str(result)}))
    return 0
