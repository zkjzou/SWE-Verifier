"""Pure slate transforms and durable JSONL stages.

The regex-token Jaccard metric and nonempty-observation deduplication follow
SWE-PRM's swesmith_provider_replacement_goldguard implementation (a8638001).
The public schema separates teacher observations from student messages.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast

SCHEMA_VERSION = 1


class Message(TypedDict, total=False):
    role: str
    content: str
    tool_call_id: str
    tool_calls: list[dict[str, object]]
    name: str


class Provenance(TypedDict):
    model: str
    revision: str
    prompt_id: str
    source_commit: str


class Observation(TypedDict):
    output: str
    returncode: int
    environment: str


class Candidate(TypedDict, total=False):
    candidate_id: str
    command: str
    content: str
    provenance: Provenance
    observation: Observation
    annotation: dict[str, object]
    message: Message


class Slate(TypedDict):
    schema_version: int
    slate_id: str
    task_id: str
    trajectory_id: str
    step_id: int
    task: str
    history: list[Message]
    prefix_commands: list[str]
    replay: dict[str, str]
    provenance: Provenance
    candidates: list[Candidate]


class Sampler(Protocol):
    def sample(self, slate: Slate) -> Sequence[Candidate]: ...


class Replayer(Protocol):
    def replay(self, slate: Slate, candidate: Candidate) -> Observation: ...


class Teacher(Protocol):
    def label(self, slate: Slate) -> Sequence[Mapping[str, object]]: ...


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            yield value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterator[Mapping[str, object]]) -> int:
    """Stream a dataset to a sibling temporary file before publishing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty string {key!r}")
    return value


def checked_history(value: object) -> list[Message]:
    if not isinstance(value, list):
        raise ValueError("history must be a list of text messages")
    result: list[Message] = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("Each history message must be an object")
        role = required_string(message, "role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError("Messages require a supported role and text content")
        normalized: Message = {"role": role, "content": content}
        for key in ("tool_call_id", "name"):
            if key in message:
                normalized[key] = required_string(message, key)
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
                raise ValueError("tool_calls must be a list of function-call objects")
            normalized["tool_calls"] = json.loads(canonical_json(calls))
        if role == "tool" and "tool_call_id" not in normalized:
            raise ValueError("A tool observation requires tool_call_id")
        result.append(normalized)
    return result


def checked_provenance(value: object) -> Provenance:
    if not isinstance(value, dict):
        raise ValueError("provenance must identify model, revision, prompt_id, and source_commit")
    return Provenance(**{key: required_string(value, key) for key in Provenance.__annotations__})


def validate_slate(row: Mapping[str, object]) -> Slate:
    """Validate a canonical slate without mutating it; reject incomplete or ambiguous IDs."""
    if type(row.get("schema_version")) is not int or row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported slate schema: {row.get('schema_version')!r}")
    for key in ("slate_id", "task_id", "trajectory_id", "task"):
        required_string(row, key)
    step_id = row.get("step_id")
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
        raise ValueError("step_id must be a nonnegative integer")
    checked_history(row.get("history"))
    checked_provenance(row.get("provenance"))
    prefix = row.get("prefix_commands")
    if not isinstance(prefix, list) or not all(isinstance(x, str) and x.strip() for x in prefix):
        raise ValueError("prefix_commands must contain nonempty shell commands")
    replay = row.get("replay")
    if not isinstance(replay, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in replay.items()):
        raise ValueError("replay must be a string mapping identifying an immutable environment")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("A slate must have at least one candidate")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be an object")
        candidate_id = required_string(candidate, "candidate_id")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        required_string(candidate, "command")
        required_string(candidate, "content")
        checked_provenance(candidate.get("provenance"))
        if "message" in candidate:
            message = checked_history([candidate["message"]])[0]
            if message["role"] != "assistant":
                raise ValueError("A candidate message must have assistant role")
        if "observation" in candidate:
            observation = candidate["observation"]
            if not isinstance(observation, dict) or not isinstance(observation.get("output"), str):
                raise ValueError("observation.output must be text")
            if type(observation.get("returncode")) is not int:
                raise ValueError("observation.returncode must be an integer")
            required_string(observation, "environment")
        if "annotation" in candidate:
            annotation = candidate["annotation"]
            if not isinstance(annotation, dict):
                raise ValueError("annotation must be an object")
            score = annotation.get("score")
            if type(score) not in (int, float) or not math.isfinite(float(score)) or not 0 <= score <= 1:
                raise ValueError("annotation.score must be a finite number in [0, 1]")
            required_string(annotation, "target_text")
            checked_provenance(annotation.get("provenance"))
    return cast(Slate, json.loads(canonical_json(row)))


def collect_successful(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand a successful trajectory into pre-action states.

    Input fields: task_id, trajectory_id, task, resolved (evaluation bool),
    provenance, replay, and steps=[{command, content, observation}]. A submitted
    patch alone is not evidence of success. Initial messages live in `history`.
    """
    if type(row.get("resolved")) is not bool:
        raise ValueError("resolved must be an explicit evaluation boolean")
    if row["resolved"] is False:
        return []
    task_id = required_string(row, "task_id")
    trajectory_id = required_string(row, "trajectory_id")
    task = required_string(row, "task")
    provenance = checked_provenance(row.get("provenance"))
    history = checked_history(row.get("history"))
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("A successful trajectory requires a nonempty steps list")
    result: list[dict[str, object]] = []
    prefix: list[str] = []
    for step_id, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError("Trajectory steps must be objects")
        command = required_string(step, "command")
        content = required_string(step, "content")
        observation = step.get("observation")
        if not isinstance(observation, dict) or not isinstance(observation.get("output"), str):
            raise ValueError("Every trajectory step needs its observed output")
        slate_id = digest([task_id, trajectory_id, step_id])
        candidate: Candidate = {
            "candidate_id": digest([slate_id, "reference"]),
            "command": command,
            "content": content,
            "provenance": provenance,
        }
        slate = {
            "schema_version": SCHEMA_VERSION,
            "slate_id": slate_id,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "step_id": step_id,
            "task": task,
            "history": list(history),
            "prefix_commands": list(prefix),
            "replay": row.get("replay"),
            "provenance": provenance,
            "candidates": [candidate],
        }
        result.append(dict(validate_slate(slate)))
        history = [
            *history,
            {"role": "assistant", "content": content},
            {"role": "user", "content": observation["output"]},
        ]
        prefix = [*prefix, command]
    return result


def sample_candidates(row: Mapping[str, object], sampler: Sampler) -> list[dict[str, object]]:
    slate = validate_slate(row)
    additions = list(sampler.sample(slate))
    updated = {**slate, "candidates": [*slate["candidates"], *additions]}
    return [dict(validate_slate(updated))]


def command_similarity(left: str, right: str) -> float:
    """SWE-PRM training regex-token Jaccard, including punctuation-only commands."""
    left_tokens = set(re.findall(r"[A-Za-z0-9_]+", left.lower())) or {left.strip().lower()}
    right_tokens = set(re.findall(r"[A-Za-z0-9_]+", right.lower())) or {right.strip().lower()}
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def filter_actions(row: Mapping[str, object], threshold: float) -> list[dict[str, object]]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Jaccard threshold must be in [0, 1]")
    slate = validate_slate(row)
    retained: list[Candidate] = []
    discarded: list[str] = []
    for candidate in slate["candidates"]:
        if any(command_similarity(candidate["command"], previous["command"]) >= threshold for previous in retained):
            discarded.append(candidate["candidate_id"])
        else:
            retained.append(candidate)
    return [{**slate, "candidates": retained, "action_filter": {"threshold": threshold, "discarded": discarded}}]


def replay_candidates(row: Mapping[str, object], replayer: Replayer) -> list[dict[str, object]]:
    slate = validate_slate(row)
    candidates = [{**candidate, "observation": replayer.replay(slate, candidate)} for candidate in slate["candidates"]]
    return [dict(validate_slate({**slate, "candidates": candidates}))]


def deduplicate_observations(row: Mapping[str, object]) -> list[dict[str, object]]:
    slate = validate_slate(row)
    retained: list[Candidate] = []
    seen: set[str] = set()
    discarded: list[str] = []
    for candidate in slate["candidates"]:
        if "observation" not in candidate:
            raise ValueError("Replay must complete before observation deduplication")
        output = candidate["observation"]["output"].strip()
        if output and output in seen:
            discarded.append(candidate["candidate_id"])
            continue
        if output:
            seen.add(output)
        retained.append(candidate)
    return [{**slate, "candidates": retained, "observation_filter": {"discarded": discarded}}]


def label_candidates(row: Mapping[str, object], teacher: Teacher) -> list[dict[str, object]]:
    slate = validate_slate(row)
    if any("observation" not in candidate for candidate in slate["candidates"]):
        raise ValueError("Teacher annotation requires replay observations for every candidate")
    labels = list(teacher.label(slate))
    expected = {candidate["candidate_id"] for candidate in slate["candidates"]}
    received = [required_string(label, "candidate_id") for label in labels]
    if set(received) != expected or len(received) != len(expected):
        raise ValueError("Teacher must annotate each candidate ID exactly once")
    by_id = {required_string(label, "candidate_id"): dict(label) for label in labels}
    candidates = [{**candidate, "annotation": by_id[candidate["candidate_id"]]} for candidate in slate["candidates"]]
    return [dict(validate_slate({**slate, "candidates": candidates}))]


def student_messages(slate: Slate, candidate: Candidate) -> list[Message]:
    """Allowlist student inputs; observations/labels/future evidence never enter this path."""
    proposed = candidate.get("message", {"role": "assistant", "content": candidate["content"]})
    return [*checked_history(slate["history"]), *checked_history([proposed])]


def run_stage(
    input_path: Path,
    output_dir: Path,
    stage: str,
    settings: Mapping[str, object],
    transform: Callable[[Mapping[str, object]], Sequence[Mapping[str, object]]],
) -> Path:
    """Resume completed input rows; reject changed inputs/settings/code and concurrent writers.

    The cache is committed after each source row. A crash inside an external call
    can repeat that one row, so provider-level exactly-once billing is not promised.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "input_sha256": file_digest(input_path),
        "pipeline_sha256": file_digest(Path(__file__)),
        "settings": dict(settings),
    }
    with (output_dir / ".lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Stage already running: {output_dir}") from exc
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Incompatible resume in {output_dir}; choose a new output directory")
        else:
            atomic_text(manifest_path, canonical_json(manifest) + "\n")
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        def completed_rows() -> Iterator[Mapping[str, object]]:
            for index, row in enumerate(read_jsonl(input_path)):
                cache_path = cache_dir / f"{index:08d}.json"
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text())
                    if cached.get("input_sha256") != digest(row) or cached.get("output_sha256") != digest(
                        cached.get("rows")
                    ):
                        raise ValueError(f"Corrupt stage cache: {cache_path}")
                    output_rows = cached["rows"]
                else:
                    output_rows = [dict(item) for item in transform(row)]
                    atomic_text(
                        cache_path,
                        canonical_json(
                            {"input_sha256": digest(row), "output_sha256": digest(output_rows), "rows": output_rows}
                        ),
                    )
                yield from output_rows

        destination = output_dir / "slates.jsonl"
        count = atomic_jsonl(destination, completed_rows())
        atomic_text(
            output_dir / "summary.json",
            canonical_json({"stage": stage, "rows": count, "sha256": file_digest(destination)}),
        )
        return destination


def task_split(task_id: str, seed: str, validation_fraction: float) -> str:
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and in [0, 1)")
    fraction = int(digest([seed, task_id])[:16], 16) / 2**64
    return "validation" if fraction < validation_fraction else "train"


def export_dataset(input_path: Path, output_dir: Path, seed: str, validation_fraction: float) -> dict[str, int]:
    """Export canonical grouped slates and candidate-local supervision with task-level splits."""
    task_split("validation", seed, validation_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"seed": seed, "validation_fraction": validation_fraction}

    def with_split(row: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
        slate = validate_slate(row)
        if any("annotation" not in candidate for candidate in slate["candidates"]):
            raise ValueError(f"Cannot export unlabeled slate {slate['slate_id']}")
        return [{**slate, "split": task_split(slate["task_id"], seed, validation_fraction)}]

    grouped = run_stage(input_path, output_dir, "export", settings, with_split)
    counts = {"train": 0, "validation": 0}
    seen: set[str] = set()

    def targets() -> Iterator[Mapping[str, object]]:
        for row in read_jsonl(grouped):
            slate = validate_slate(row)
            if slate["slate_id"] in seen:
                raise ValueError(f"Duplicate slate_id at export: {slate['slate_id']}")
            seen.add(slate["slate_id"])
            split = cast(str, row["split"])
            for candidate in slate["candidates"]:
                annotation = candidate["annotation"]
                target = {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": slate["task_id"],
                    "trajectory_id": slate["trajectory_id"],
                    "step_id": slate["step_id"],
                    "slate_id": slate["slate_id"],
                    "candidate_id": candidate["candidate_id"],
                    "split": split,
                    "task": slate["task"],
                    "messages": student_messages(slate, candidate),
                    "target_text": annotation["target_text"],
                    "score": annotation["score"],
                    "rubric": annotation.get("rubric", ""),
                    "summary": annotation.get("summary", ""),
                    "provenance": candidate["provenance"],
                    "teacher_provenance": annotation["provenance"],
                }
                yield target
                counts[split] += 1

    atomic_jsonl(output_dir / "per_action.jsonl", targets())
    atomic_text(
        output_dir / "dataset_info.json",
        canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "split_unit": "task_id",
                "seed": seed,
                "validation_fraction": validation_fraction,
                "slates": len(seen),
                "actions": counts,
                "slates_sha256": file_digest(grouped),
                "per_action_sha256": file_digest(output_dir / "per_action.jsonl"),
            }
        )
        + "\n",
    )
    return counts
