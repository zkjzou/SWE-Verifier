"""Import Mini-swe-agent trajectories using explicit external success labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from swe_verifier.data.pipeline import (
    SCHEMA_VERSION,
    Candidate,
    checked_history,
    checked_provenance,
    digest,
    read_jsonl,
    required_string,
    validate_slate,
)


def _trajectory_path(row: Mapping[str, object], base_dir: Path) -> Path:
    path = Path(required_string(row, "trajectory_path"))
    return path if path.is_absolute() else base_dir / path


def source_fingerprints(index_path: Path) -> dict[str, str]:
    """Include referenced files in resume compatibility, not just the index text."""
    result: dict[str, str] = {}
    for row in read_jsonl(index_path):
        trajectory_id = required_string(row, "trajectory_id")
        if trajectory_id in result:
            raise ValueError(f"Duplicate trajectory_id in import index: {trajectory_id}")
        result[trajectory_id] = hashlib.sha256(_trajectory_path(row, index_path.parent).read_bytes()).hexdigest()
    return result


def collect_mini_index(row: Mapping[str, object], base_dir: Path) -> list[dict[str, object]]:
    """Convert one Mini trajectory-index entry to canonical, pre-action slates.

    Index metadata supplies trajectory_path, task_id, trajectory_id, task,
    resolved, replay={image,workdir}, and provenance. Resolution must come from
    the evaluation harness, never the trajectory's submission/exit status.
    """
    if type(row.get("resolved")) is not bool:
        raise ValueError("resolved must be an explicit external evaluation boolean")
    if row["resolved"] is False:
        return []
    task_id = required_string(row, "task_id")
    trajectory_id = required_string(row, "trajectory_id")
    task = required_string(row, "task")
    provenance = checked_provenance(row.get("provenance"))
    payload = json.loads(_trajectory_path(row, base_dir).read_text())
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        raise ValueError("Mini trajectory must contain a messages list")
    history: list[dict[str, object]] = []
    prefix: list[str] = []
    result: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Mini trajectory contains a non-object message")
        if message.get("role") == "exit":
            continue
        current = {**message, "content": message.get("content") or ""}
        normalized = checked_history([current])[0]
        extra = message.get("extra")
        actions = extra.get("actions") if isinstance(extra, dict) else None
        if message.get("role") == "assistant" and actions:
            if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
                raise ValueError("Mini data import requires one action per step; parallel actions are unsupported")
            command = required_string(actions[0], "command")
            slate_id = digest([task_id, trajectory_id, len(prefix)])
            content = normalized["content"] or f"```bash\n{command}\n```"
            candidate: Candidate = {
                "candidate_id": digest([slate_id, "reference"]),
                "command": command,
                "content": content,
                "message": normalized,
                "provenance": provenance,
            }
            slate = {
                "schema_version": SCHEMA_VERSION,
                "slate_id": slate_id,
                "task_id": task_id,
                "trajectory_id": trajectory_id,
                "step_id": len(prefix),
                "task": task,
                "history": history,
                "prefix_commands": prefix,
                "replay": row.get("replay"),
                "provenance": provenance,
                "candidates": [candidate],
            }
            result.append(dict(validate_slate(slate)))
            prefix = [*prefix, command]
        history = [*history, dict(normalized)]
    if not result:
        raise ValueError("Successful Mini trajectory has no parsed actions in message.extra.actions")
    return result
