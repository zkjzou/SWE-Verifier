"""Construct a tiny synthetic dataset with real local command execution; no model APIs."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from swe_verifier.data.adapters import ModelSampler, ModelTeacher, SnapshotReplay
from swe_verifier.data.pipeline import (
    Provenance,
    collect_successful,
    deduplicate_observations,
    export_dataset,
    filter_actions,
    label_candidates,
    read_jsonl,
    replay_candidates,
    run_stage,
    sample_candidates,
)


class FixtureModel:
    def __init__(self, responses: Sequence[dict[str, object]]):
        self.responses = iter(responses)

    def query(self, messages: list[dict[str, object]]) -> dict[str, object]:
        if not messages:
            raise ValueError("Expected a rendered model prompt")
        return next(self.responses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    provenance: Provenance = {
        "model": "synthetic-fixture",
        "revision": "v1",
        "prompt_id": "synthetic-policy-v1",
        "source_commit": "synthetic-fixture-v1",
    }
    sampled_responses = list(read_jsonl(fixtures / "sampler-responses.jsonl"))
    teacher_response = (fixtures / "teacher-response.txt").read_text()
    sampler = ModelSampler(FixtureModel(sampled_responses), provenance, len(sampled_responses))
    teacher = ModelTeacher(FixtureModel([{"role": "assistant", "content": teacher_response}]), provenance)
    settings = {"fixture": "synthetic-v1"}
    collected = run_stage(
        fixtures / "trajectories.jsonl", args.output / "01-collect", "collect", settings, collect_successful
    )
    sampled = run_stage(
        collected, args.output / "02-sample", "sample", settings, functools.partial(sample_candidates, sampler=sampler)
    )
    filtered = run_stage(
        sampled,
        args.output / "03-filter",
        "filter",
        {"threshold": 0.8},
        functools.partial(filter_actions, threshold=0.8),
    )
    replay_settings = {"snapshot_sha256": hashlib.sha256((fixtures / "snapshot/README.txt").read_bytes()).hexdigest()}
    replayed = run_stage(
        filtered,
        args.output / "04-replay",
        "replay",
        replay_settings,
        functools.partial(replay_candidates, replayer=SnapshotReplay(fixtures / "snapshot", 10)),
    )
    deduped = run_stage(replayed, args.output / "05-deduplicate", "deduplicate", settings, deduplicate_observations)
    labeled = run_stage(
        deduped, args.output / "06-label", "label", settings, functools.partial(label_candidates, teacher=teacher)
    )
    result = export_dataset(labeled, args.output / "final", "synthetic-v1", 0.1)
    print(json.dumps({"synthetic_fixture": True, "actions": result, "output": str(args.output / "final")}))


if __name__ == "__main__":
    main()
