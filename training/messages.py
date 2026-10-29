"""Native Qwen tool-argument normalization from the research runtime."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

Qwen35MessageContractError = ValueError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Qwen35MessageContractError(message)


def normalize_qwen35_chat_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI JSON-string tool arguments for Qwen3.5's native Jinja."""

    normalized = copy.deepcopy([dict(message) for message in messages])
    for message_index, message in enumerate(normalized):
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        _require(isinstance(tool_calls, list), f"message {message_index} tool_calls is not a list")
        for tool_index, tool_call in enumerate(tool_calls):
            _require(isinstance(tool_call, dict), f"message {message_index} tool call {tool_index} is not an object")
            function = tool_call.get("function") if "function" in tool_call else tool_call
            _require(
                isinstance(function, dict),
                f"message {message_index} tool call {tool_index} has no function mapping",
            )
            _require(bool(str(function.get("name") or "").strip()), f"message {message_index} tool call has no name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise Qwen35MessageContractError(
                        f"message {message_index} tool call {tool_index} has invalid JSON arguments"
                    ) from exc
            _require(
                isinstance(arguments, Mapping),
                f"message {message_index} tool call {tool_index} arguments are not a mapping",
            )
            function["arguments"] = dict(arguments)
    return normalized
