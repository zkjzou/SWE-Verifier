"""End-to-end fixture construction exercises real filesystem replay and durable resume."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from swe_verifier.data.adapters import ModelSampler, ModelTeacher, SnapshotReplay, parse_teacher
from swe_verifier.data.importers import collect_mini_index, source_fingerprints
from swe_verifier.data.pipeline import (
    Candidate,
    Provenance,
    collect_successful,
    command_similarity,
    deduplicate_observations,
    export_dataset,
    filter_actions,
    label_candidates,
    read_jsonl,
    replay_candidates,
    run_stage,
    sample_candidates,
    student_messages,
    validate_slate,
)

PROVENANCE: Provenance = {
    "model": "fixture-model",
    "revision": "fixture-v1",
    "prompt_id": "fixture-policy",
    "source_commit": "a8638001",
}


class RecordedModel:
    """Deterministic fixture connector, deliberately preserving actual response shape."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.inputs: list[list[dict[str, object]]] = []

    def query(self, messages: list[dict[str, object]]) -> dict[str, object]:
        self.inputs.append(messages)
        return {"role": "assistant", "content": next(self.responses)}


def trajectory(task_id: str, trajectory_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "task": "Inspect the repository",
        "resolved": True,
        "provenance": dict(PROVENANCE),
        "replay": {"image": "fixture-image", "workdir": "/testbed"},
        "history": [
            {"role": "system", "content": "Return one bash command."},
            {"role": "user", "content": "Inspect the repository"},
        ],
        "steps": [
            {
                "command": "printf same",
                "content": "```bash\nprintf same\n```",
                "observation": {"output": "same", "returncode": 0},
            }
        ],
    }


def teacher_text(count: int) -> str:
    blocks = [
        f"- Candidate {index}:\n  NEXT_STATES_PREDICTION:\n"
        "    Immediate_Observation: predicted output.\n    Expected_Progress: inspection.\n"
        "  CHECKLIST_ITEM_SCORES:\n  - Item 1: 0.8\n    REASONING: inspection advances task.\n"
        "  REASONING: This action exposes repository state.\n  SCORE: 0.8"
        for index in range(1, count + 1)
    ]
    return (
        "<summary>Task state</summary>\n<checklist>1. Inspect files [weight: must]</checklist>\n"
        + "\n".join(blocks)
        + "\nFINAL: 1\n"
    )


def test_complete_pipeline_resume_splits_and_teacher_boundary(tmp_path: Path) -> None:
    source = tmp_path / "trajectories.jsonl"
    source.write_text(
        "\n".join(json.dumps(trajectory("same-task", name)) for name in ("trajectory-a", "trajectory-b")) + "\n"
    )
    collected = run_stage(source, tmp_path / "collect", "collect", {}, collect_successful)
    rows = list(read_jsonl(collected))
    assert len(rows) == 2
    commands = ["printf same", "printf different", "printf '%s' same", "true", ":"]
    responses = [f"```bash\n{command}\n```" for command in commands] * 2
    sampler = ModelSampler(RecordedModel(responses), PROVENANCE, len(commands))
    sampled = run_stage(
        collected,
        tmp_path / "sample",
        "sample",
        {"model": "fixture"},
        functools.partial(sample_candidates, sampler=sampler),
    )
    # Resume must not consume the exhausted fixture model.
    assert (
        run_stage(
            collected,
            tmp_path / "sample",
            "sample",
            {"model": "fixture"},
            functools.partial(sample_candidates, sampler=sampler),
        )
        == sampled
    )
    filtered = run_stage(
        sampled, tmp_path / "filter", "filter", {"threshold": 0.8}, functools.partial(filter_actions, threshold=0.8)
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "state").write_text("untouched")
    replayed = run_stage(
        filtered,
        tmp_path / "replay",
        "replay",
        {"fixture": "snapshot"},
        functools.partial(replay_candidates, replayer=SnapshotReplay(snapshot, 10)),
    )
    deduplicated = run_stage(replayed, tmp_path / "deduplicate", "deduplicate", {}, deduplicate_observations)
    dedup_rows = list(read_jsonl(deduplicated))
    assert len(dedup_rows[0]["candidates"]) == 4  # identical nonempty output removed; empty outputs retained
    model = RecordedModel([teacher_text(4), teacher_text(4)])
    labeled = run_stage(
        deduplicated,
        tmp_path / "label",
        "label",
        {"teacher": "fixture"},
        functools.partial(label_candidates, teacher=ModelTeacher(model, PROVENANCE)),
    )
    assert "Immediate observation:" in str(model.inputs[0])
    counts = export_dataset(labeled, tmp_path / "export", "test-seed", 0.5)
    exported = list(read_jsonl(tmp_path / "export" / "per_action.jsonl"))
    assert sum(counts.values()) == len(exported) == 8
    assert len({row["split"] for row in exported}) == 1  # same task cannot span splits
    for row in exported:
        assert set(row["messages"][-1]) == {"role", "content"}
        assert row["messages"][-1]["role"] == "assistant"
        assert "observation" not in row
        assert "raw_teacher_output" not in row
        assert "Immediate observation:" not in json.dumps(row["messages"])
    assert (snapshot / "state").read_text() == "untouched"


def test_replay_starts_each_candidate_from_same_prefix(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "state").write_text("initial")
    slate = validate_slate(collect_successful(trajectory("task", "run"))[0])
    slate["prefix_commands"] = ["printf prefix > state"]
    replayer = SnapshotReplay(snapshot, 10)
    first: Candidate = {
        "candidate_id": "a",
        "command": "printf changed > state; cat state",
        "content": "edit",
        "provenance": PROVENANCE,
    }
    second: Candidate = {"candidate_id": "b", "command": "cat state", "content": "inspect", "provenance": PROVENANCE}
    assert replayer.replay(slate, first)["output"] == "changed"
    assert replayer.replay(slate, second)["output"] == "prefix"
    assert (snapshot / "state").read_text() == "initial"


def test_resume_rejects_changed_settings_and_corrupt_cache(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(trajectory("task", "run")) + "\n")
    output = tmp_path / "stage"
    run_stage(source, output, "collect", {"version": 1}, collect_successful)
    with pytest.raises(ValueError, match="Incompatible resume"):
        run_stage(source, output, "collect", {"version": 2}, collect_successful)
    cache = output / "cache" / "00000000.json"
    stored = json.loads(cache.read_text())
    stored["rows"] = []
    cache.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="Corrupt stage cache"):
        run_stage(source, output, "collect", {"version": 1}, collect_successful)


def test_filter_boundary_invalid_labels_and_success_contract() -> None:
    assert command_similarity("a b c d", "a b c d e") == 0.8
    slate = validate_slate(collect_successful(trajectory("task", "run"))[0])
    slate["candidates"] = [
        {"candidate_id": "a", "command": "a b c d", "content": "a", "provenance": PROVENANCE},
        {"candidate_id": "b", "command": "a b c d e", "content": "b", "provenance": PROVENANCE},
    ]
    assert len(filter_actions(slate, 0.8)[0]["candidates"]) == 1
    with pytest.raises(ValueError, match="one numbered block"):
        parse_teacher(teacher_text(1), slate, PROVENANCE)
    with pytest.raises(ValueError, match="outside"):
        parse_teacher(teacher_text(2).replace("SCORE: 0.8", "SCORE: 1.5"), slate, PROVENANCE)
    source = trajectory("task", "run")
    source["resolved"] = "true"
    with pytest.raises(ValueError, match="evaluation boolean"):
        collect_successful(source)
    source["resolved"] = False
    assert collect_successful(source) == []


def test_native_mini_import_preserves_tool_history(tmp_path: Path) -> None:
    path = tmp_path / "mini.json"
    payload = {
        "messages": [
            {"role": "user", "content": "Inspect repository"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}
                ],
                "extra": {"actions": [{"command": "pwd"}]},
            },
            {"role": "tool", "content": "/testbed", "tool_call_id": "call1"},
            {"role": "assistant", "content": "```bash\nls\n```", "extra": {"actions": [{"command": "ls"}]}},
            {"role": "exit", "content": "Submitted"},
        ]
    }
    path.write_text(json.dumps(payload))
    index_row = {key: value for key, value in trajectory("task", "run").items() if key not in {"history", "steps"}}
    index_row["trajectory_path"] = path.name
    rows = collect_mini_index(index_row, tmp_path)
    assert len(rows) == 2
    assert rows[0]["history"] == [{"role": "user", "content": "Inspect repository"}]
    assert rows[1]["history"][-1]["tool_call_id"] == "call1"
    assert rows[1]["prefix_commands"] == ["pwd"]
    labeled_slate = validate_slate(rows[1])
    labeled_slate["candidates"][0]["observation"] = {
        "output": "teacher-only-current-observation",
        "returncode": 0,
        "environment": "fixture-v1",
    }
    teacher_model = RecordedModel([teacher_text(1)])
    ModelTeacher(teacher_model, PROVENANCE).label(labeled_slate)
    assert "pwd" in str(teacher_model.inputs[0])
    assert "teacher-only-current-observation" in str(teacher_model.inputs[0])
    student = student_messages(labeled_slate, labeled_slate["candidates"][0])
    assert "teacher-only-current-observation" not in str(student)
    assert student[1]["tool_calls"][0]["id"] == "call1"
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(index_row) + "\n")
    before = source_fingerprints(index)
    path.write_text(json.dumps(payload, indent=2))
    assert source_fingerprints(index) != before
