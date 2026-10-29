"""Vendored unmodified campaign parsers and native tool-call normalizer.

Source: minisweagent.utils.qwen35_checklist_modes_data and qwen35_messages.
Only imports and the shared error helper are simplified for standalone use.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

Qwen35MessageContractError = ValueError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_generator_response(text: str) -> dict[str, Any]:
    """Check one complete checklist; malformed checklists never become dependencies."""
    errors: list[str] = []
    body, ids, criteria = None, [], []
    match = re.fullmatch(r"\s*<checklist>\n(?P<body>.*?)\n</checklist>\s*", str(text), re.DOTALL)
    if match is None or len(re.findall(r"(?m)^</?checklist>$", str(text))) != 2:
        errors.append("checklist_delimiters_or_extra_output")
    else:
        body = match["body"].strip()
        intro = re.match(r"REASONING:[ \t]*(\S.*?)\n(?=1\. )", body, re.DOTALL)
        if intro is None:
            errors.append("checklist_introductory_reasoning")
        items_text = body[intro.end() :] if intro else body
        pattern = re.compile(r"(?m)^(\d+)\.[ \t]+([^\n]+?)[ \t]+\[weight:[ \t]*(must|should|nice)\][ \t]*(?:\n|\Z)")
        matches = list(pattern.finditer(items_text))
        ids = [int(item[1]) for item in matches]
        criteria = [{"id": int(item[1]), "criterion": item[2].strip(), "weight": item[3]} for item in matches]
        if not ids or ids != list(range(1, len(ids) + 1)):
            errors.append("checklist_item_ids_or_weights")
        if pattern.sub("", items_text).strip():
            errors.append("checklist_unparsed_content")
        if re.search(r"(?m)^</?(?:summary|prediction|checklist_scores|reasoning|checklist)>|^SCORE:|^FINAL:", body):
            errors.append("checklist_forbidden_output")
    return {
        "format_valid": not errors,
        "errors": sorted(set(errors)),
        "checklist_text": body if not errors else None,
        "checklist_items": len(ids),
        "item_ids": ids,
        "criteria": criteria,
    }


def checklist_body(checklist: str) -> str:
    """Validate supplied criteria without evaluating any Jinja embedded in the text."""
    _require(isinstance(checklist, str) and bool(checklist.strip()), "supplied checklist is empty")
    text = checklist.strip()
    parsed = parse_generator_response(text if text.startswith("<checklist>") else f"<checklist>\n{text}\n</checklist>")
    _require(parsed["format_valid"], f"invalid supplied checklist: {parsed['errors']}")
    return parsed["checklist_text"]


def _terminal_score(text: str) -> tuple[float | None, list[str]]:
    errors = []
    score = None
    match = re.search(r"(?:^|\n)SCORE:\s*([^\s]+)\s*\Z", text)
    if match:
        try:
            value = float(match[1])
            if math.isfinite(value):
                score = value
        except ValueError:
            pass
    if score is None:
        errors.append("missing_finite_terminal_score")
    elif not 0 <= score <= 1:
        errors.append("terminal_score_range")
    elif not re.fullmatch(r"(?:0\.\d{2}|1\.00)", match[1]):
        errors.append("terminal_score_precision")
    if len(re.findall(r"(?m)^SCORE:", text)) != 1:
        score = None
        errors.append("multiple_or_missing_score_lines")
    return score, errors


def parse_scorer_response(text: str, checklist: str) -> dict[str, Any]:
    """Expose raw finite SCORE independently from output-format diagnostics."""
    body = checklist_body(checklist)
    expected = parse_generator_response(f"<checklist>\n{body}\n</checklist>")["item_ids"]
    score, errors = _terminal_score(text)
    match = re.fullmatch(
        r"\s*<checklist_scores>\n(?P<items>.*?)\n</checklist_scores>\n"
        r"REASONING:[ \t]*(?P<reason>\S.*?)\nSCORE:[ \t]*[^\s]+\s*",
        text,
        re.DOTALL,
    )
    items, reason = [], ""
    if match:
        pattern = re.compile(r"(?ms)^- Item (\d+):[ \t]*([^\s]+)[ \t]*\n[ \t]+REASONING:[ \t]*(.*?)(?=^- Item |\Z)")
        items = pattern.findall(match["items"])
        reason = match["reason"].strip()
        if pattern.sub("", match["items"]).strip():
            errors.append("checklist_scores_unparsed_content")
        if re.search(r"(?i)\b(?:Immediate_Observation|Expected_Progress|Item[ \t]+\d+)\b", reason):
            errors.append("reasoning_not_natural_prose")
    else:
        errors.append("scorer_section_order_or_delimiters")
    if [int(item[0]) for item in items] != expected:
        errors.append("checklist_score_alignment")
    explanations = 0
    for _, raw, explanation in items:
        if not re.fullmatch(r"(?:0\.\d{2}|1\.00)", raw):
            errors.append("checklist_score_range_or_precision")
        if explanation.strip() and explanation.strip() not in ("None", "null", "..."):
            explanations += 1
        else:
            errors.append("missing_item_explanation")
    if len(re.findall(r"(?m)^</?checklist_scores>$", text)) != 2:
        errors.append("duplicate_or_missing_checklist_scores_tag")
    if re.search(r"(?m)^</?(?:summary|prediction|checklist|reasoning)>|^Candidate\s+\d+\s*:|^FINAL:", text):
        errors.append("scorer_forbidden_output")
    return {
        "score": score,
        "format_valid": not errors,
        "errors": sorted(set(errors)),
        "checklist_items": len(expected),
        "checklist_score_items": len(items),
        "explanation_items": explanations,
        "reasoning": reason,
    }


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
