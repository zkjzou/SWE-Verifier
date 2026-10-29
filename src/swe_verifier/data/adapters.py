"""External-system adapters for policy sampling, teacher calls, and isolated replay."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from swe_verifier.data.pipeline import Candidate, Observation, Provenance, Slate, checked_history, digest
from swe_verifier.prompts import render


class TextModel(Protocol):
    def query(self, messages: list[dict[str, object]]) -> dict[str, object]: ...


class ModelSampler:
    """Sample one model from the unmodified observed policy prefix."""

    def __init__(self, model: TextModel, provenance: Provenance, samples: int):
        if samples < 1:
            raise ValueError("samples must be positive")
        self.model = model
        self.provenance = dict(provenance)
        self.samples = samples

    def sample(self, slate: Slate) -> Sequence[Candidate]:
        candidates: list[Candidate] = []
        messages = [dict(message) for message in slate["history"]]
        if not messages:
            raise ValueError("Sampling requires the original policy system/task history")
        for index in range(self.samples):
            response = self.model.query(messages)
            content = response.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Sampler returned unsupported nontext content")
            extra = response.get("extra")
            actions = extra.get("actions") if isinstance(extra, dict) else None
            if isinstance(actions, list):
                if len(actions) != 1 or not isinstance(actions[0], dict):
                    raise ValueError("Sampling requires exactly one bash action")
                command = actions[0].get("command")
                if not isinstance(command, str) or not command.strip():
                    raise ValueError("Sampler action lacks a command")
            else:
                commands = re.findall(r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```", content, re.DOTALL)
                if len(commands) != 1 or not commands[0].strip():
                    raise ValueError("Sampler must return exactly one bash code block")
                command = commands[0]
            candidate: Candidate = {
                "candidate_id": digest([slate["slate_id"], self.provenance, index]),
                "command": command,
                "content": content or f"```bash\n{command}\n```",
                "provenance": self.provenance,
                "message": checked_history([{**response, "content": content}])[0],
            }
            candidates.append(candidate)
        return candidates


class CombinedSampler:
    def __init__(self, samplers: Sequence[ModelSampler]):
        if not samplers:
            raise ValueError("At least one sampler is required")
        self.samplers = tuple(samplers)

    def sample(self, slate: Slate) -> Sequence[Candidate]:
        return [candidate for sampler in self.samplers for candidate in sampler.sample(slate)]


def parse_teacher(text: str, slate: Slate, provenance: Provenance) -> list[dict[str, object]]:
    """Parse paper teacher blocks while retaining raw output for audit."""
    summary = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    rubric = re.search(r"<checklist>(.*?)</checklist>", text, re.DOTALL)
    if summary is None or rubric is None:
        raise ValueError("Teacher output must contain summary and checklist sections")
    pattern = re.compile(r"(?m)^\s*-?\s*Candidate\s+(\d+)\s*:\s*$")
    headers = list(pattern.finditer(text))
    if len(headers) != len(slate["candidates"]):
        raise ValueError("Teacher output must include one numbered block per candidate")
    labels: list[dict[str, object]] = []
    seen: set[int] = set()
    for position, header in enumerate(headers):
        number = int(header.group(1))
        if not 1 <= number <= len(slate["candidates"]) or number in seen:
            raise ValueError(f"Invalid or duplicate teacher candidate number: {number}")
        seen.add(number)
        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        block = text[header.end() : end].strip()
        scores = list(re.finditer(r"(?m)^\s*SCORE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", block))
        if len(scores) != 1:
            raise ValueError(f"Teacher candidate {number} requires exactly one SCORE field")
        score = float(scores[0].group(1))
        if not 0 <= score <= 1:
            raise ValueError(f"Teacher score outside [0, 1]: {score}")
        target = block[: scores[0].end()].strip()
        if "NEXT_STATES_PREDICTION:" not in target or "REASONING:" not in target:
            raise ValueError(f"Teacher candidate {number} lacks prediction/reasoning supervision")
        labels.append(
            {
                "candidate_id": slate["candidates"][number - 1]["candidate_id"],
                "score": score,
                "target_text": target,
                "summary": summary.group(1).strip(),
                "rubric": rubric.group(1).strip(),
                "raw_teacher_output": text,
                "provenance": dict(provenance),
            }
        )
    return labels


class ModelTeacher:
    def __init__(self, model: TextModel, provenance: Provenance):
        self.model = model
        self.provenance = dict(provenance)

    def label(self, slate: Slate) -> Sequence[Mapping[str, object]]:
        candidates = [
            {
                "index": index,
                "content": candidate["content"],
                "next_state_observed_text": candidate["observation"]["output"],
                "future_steps_text": "",
                "future_steps_summary_text": "",
            }
            for index, candidate in enumerate(slate["candidates"])
        ]
        # The paper's flattened-history template renders content, not tool_calls.
        # Preserve native historical commands in this teacher-only text projection.
        steps: list[list[dict[str, object]]] = []
        for message in slate["history"]:
            visible = dict(message)
            if message.get("tool_calls"):
                visible["content"] = (
                    str(message.get("content") or "") + "\nTOOL_CALLS:\n" + json.dumps(message["tool_calls"])
                )
            steps.append([visible])
        variables: dict[str, object] = {
            "task": slate["task"],
            "candidates": candidates,
            "selection_index_base": 1,
            "history_message_format": "single_prompt",
            "history_steps": -1,
            "all_steps": steps,
            "steps": steps,
            "future_horizon": 0,
        }
        messages: list[dict[str, object]] = [
            {"role": "system", "content": render("teacher/system.txt", variables)},
            {"role": "user", "content": render("teacher/annotate.jinja", variables)},
        ]
        response = self.model.query(messages)
        content = response.get("content")
        if not isinstance(content, str):
            raise ValueError("Teacher returned no text output")
        return parse_teacher(content, slate, self.provenance)


class DockerReplay:
    """Replay each candidate in a fresh container without host mounts or network access.

    `replay.image` identifies a prebuilt task image; `replay.workdir` is its repository
    directory. The image is resolved to its local immutable ID before every run.
    Nonzero command exits are observations, whereas missing images/container errors
    raise explicitly. Prefix commands run in order before the proposed candidate.
    """

    def __init__(self, executable: str, timeout: int, image_revisions: Mapping[str, str]):
        if timeout < 1:
            raise ValueError("Replay timeout must be positive")
        self.executable = executable
        self.timeout = timeout
        self.image_revisions = dict(image_revisions)

    def replay(self, slate: Slate, candidate: Candidate) -> Observation:
        image = slate["replay"].get("image")
        workdir = slate["replay"].get("workdir")
        if not image or not workdir or not workdir.startswith("/"):
            raise ValueError("Docker replay requires image and absolute workdir in each slate")
        image_id = self.image_revisions.get(image)
        if not image_id or not image_id.startswith("sha256:"):
            raise ValueError(f"Image must be resolved before replay: {image!r}")
        started = subprocess.run(
            [
                self.executable,
                "run",
                "--detach",
                "--network",
                "none",
                "--workdir",
                workdir,
                image_id,
                "sleep",
                str(self.timeout * (len(slate["prefix_commands"]) + 2)),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=self.timeout,
        )
        container = started.stdout.strip()
        if not container:
            raise RuntimeError("Docker did not return a container ID")
        try:
            result: subprocess.CompletedProcess[str] | None = None
            for command in [*slate["prefix_commands"], candidate["command"]]:
                result = subprocess.run(
                    [
                        self.executable,
                        "exec",
                        "--workdir",
                        workdir,
                        container,
                        "bash",
                        "-lc",
                        f"cd {shlex.quote(workdir)} && {command}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=self.timeout,
                )
            if result is None:
                raise RuntimeError("No candidate command executed")
            return {"output": result.stdout, "returncode": result.returncode, "environment": image_id}
        finally:
            subprocess.run(
                [self.executable, "rm", "--force", container],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )


class SnapshotReplay:
    """Fresh directory copies for trusted local smoke fixtures; NOT a security sandbox.

    Never use this connector for untrusted generated commands. The production CLI
    exposes DockerReplay, which executes without mounting the host filesystem.
    """

    def __init__(self, snapshot: Path, timeout: int):
        if not snapshot.is_dir() or timeout < 1:
            raise ValueError("SnapshotReplay requires an existing directory and positive timeout")
        self.snapshot = snapshot.resolve()
        self.timeout = timeout

    def replay(self, slate: Slate, candidate: Candidate) -> Observation:
        with tempfile.TemporaryDirectory(prefix="swe-verifier-replay-") as directory:
            workspace = Path(directory) / "workspace"
            shutil.copytree(self.snapshot, workspace)
            result: subprocess.CompletedProcess[str] | None = None
            for command in [*slate["prefix_commands"], candidate["command"]]:
                result = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-c", command],
                    cwd=workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=self.timeout,
                )
            if result is None:
                raise RuntimeError("No candidate command executed")
            inventory = [
                (str(path.relative_to(self.snapshot)), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(self.snapshot.rglob("*"))
                if path.is_file()
            ]
            return {
                "output": result.stdout,
                "returncode": result.returncode,
                "environment": "trusted-fixture:" + digest(inventory),
            }
