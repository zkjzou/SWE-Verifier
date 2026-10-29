"""Project observed chat history without feeding verifier metadata back to models."""
from __future__ import annotations

import copy
import json
import re
from typing import cast

from swe_verifier.inference.types import Candidate, InvalidOutput, Message

_NATIVE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


def policy_message(message: Message) -> Message:
    """Retain provider-native history without Mini's logs or raw tool output."""
    return {key: copy.deepcopy(value) for key, value in message.items() if key != "extra"}


def native_message(message: Message) -> Message:
    return {key: copy.deepcopy(message[key]) for key in _NATIVE_KEYS if key in message}


def command(candidate: Candidate) -> str:
    extra = candidate.message.get("extra")
    if not isinstance(extra, dict) or extra.get("parse_error"):
        raise InvalidOutput(f"Candidate {candidate.id} has no valid parsed action")
    actions = extra.get("actions")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        raise InvalidOutput(f"Candidate {candidate.id} must contain exactly one executable action")
    value = actions[0].get("command")
    if not isinstance(value, str) or not value.strip():
        raise InvalidOutput(f"Candidate {candidate.id} command must be nonempty text")
    return value


def native_candidate(candidate: Candidate) -> Message:
    value = command(candidate)
    result = native_message(candidate.message)
    result["role"] = "assistant"
    if not result.get("tool_calls"):
        result["tool_calls"] = [{
            "id": f"candidate-{candidate.id}", "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": value})},
        }]
    return result


def observed_history(history: list[Message]) -> list[Message]:
    """Convert completed text or native actions to assistant/tool pairs.

    Initial system/task messages and policy format-repair turns are omitted.
    A dangling unexecuted assistant turn is never accepted as observed evidence.
    """
    result: list[Message] = []
    pending: Message | None = None
    pending_text_action = False
    for index, message in enumerate(history):
        role = message.get("role")
        if role == "assistant":
            if message.get("tool_calls"):
                pending = native_message(message)
                pending_text_action = False
            else:
                extra = message.get("extra", {})
                if not isinstance(extra, dict) or not extra.get("actions"):
                    pending = None
                    continue
                pending = native_candidate(Candidate(str(index), message))
                pending_text_action = True
        elif role in {"tool", "user"} and pending is not None:
            if role == "user" and not pending_text_action:
                # A tool-call format-correction turn is not an execution result.
                pending = None
                continue
            calls = cast(list[dict[str, object]], pending["tool_calls"])
            if len(calls) != 1:
                raise InvalidOutput("Observed history must contain one tool call per action")
            call_id = calls[0].get("id")
            observed_id = message.get("tool_call_id")
            if role == "tool" and observed_id is not None and call_id != observed_id:
                raise InvalidOutput("Observed tool result does not match its action")
            observation = native_message(message)
            observation.update(role="tool", tool_call_id=call_id)
            result.extend((pending, observation))
            pending = None
        elif role != "system":
            pending = None
    return result


def candidate_variables(candidate: Candidate, index: int) -> dict[str, object]:
    content = str(candidate.message.get("content") or "")
    action = command(candidate)
    if candidate.message.get("tool_calls") or action not in content:
        content = "\n\n".join(part for part in (content, "```\n" + action + "\n```") if part)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"^(?:[ \t]*\r?\n)+", "", content)
    return {
        "index": index, "content": content,
        "action": command(candidate), "actions": [{"command": command(candidate)}],
        "parse_error": None, "tool_calls": native_candidate(candidate).get("tool_calls"),
    }
