"""Candidate-local scoring and independent slate selection."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import cast

from jinja2 import StrictUndefined, Template

from swe_verifier.inference.checklist_contract import parse_generator_response, parse_scorer_response
from swe_verifier.inference.messages import candidate_variables, native_candidate, observed_history
from swe_verifier.inference.research_parsing import REWARD_REGEX, SELECTION_REGEX, parse_reward, parse_selection
from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import (
    Candidate,
    InvalidOutput,
    Message,
    OutputBudgetModel,
    QueryModel,
    Role,
    Score,
    Variant,
)
from swe_verifier.prompts import load_prompt


@dataclass(frozen=True)
class _RetryModel:
    """A connector view increasing only this corrective request's output budget."""
    model: OutputBudgetModel

    def query(self, messages: list[Message]) -> Message:
        return self.model.query_with_output_limit(messages, 4096)


def _two_stage_model(model: QueryModel, attempt: int) -> QueryModel:
    return _RetryModel(model) if attempt > 0 and isinstance(model, OutputBudgetModel) else model


def terminal_score(text: str) -> float:
    """Accept one terminal scalar, never a number in the model's reasoning."""
    matches = re.findall(r"(?m)^(?:SCORE|REWARD|FINAL):[ \t]*([^\s]+)[ \t]*$", text.strip())
    if len(matches) != 1 or not re.search(r"(?:^|\n)(?:SCORE|REWARD|FINAL):[^\n]+\s*\Z", text):
        raise InvalidOutput("Expected exactly one terminal SCORE, REWARD, or FINAL scalar")
    try:
        value = float(matches[0])
    except ValueError as exc:
        raise InvalidOutput(f"Invalid scalar score: {matches[0]!r}") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise InvalidOutput(f"Score must be finite and in [0, 1], received {value}")
    return value


def _text(response: Message) -> str:
    content = response.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InvalidOutput("Verifier returned empty or non-text content")
    return content


def _render(path: str, variables: dict[str, object]) -> str:
    return Template(load_prompt(path), undefined=StrictUndefined).render(**variables)


def _prefix(task: str, history: list[Message], candidate: Candidate, system: str) -> list[Message]:
    if not task.strip():
        raise ValueError("Verifier task must be nonempty")
    proposed = native_candidate(candidate)
    proposed["content"] = candidate_variables(candidate, 0)["content"]
    return [
        {"role": "system", "content": system}, {"role": "user", "content": task},
        *observed_history(history), proposed,
    ]


class Scorer:
    """Connect a model to one released prompt variant, averaging independent draws.

    Two-stage rubrics are generated afresh for every candidate and every draw.
    No proposed-action execution observation is accepted by this interface.
    """

    def __init__(self, model: QueryModel, model_name: str, variant: Variant, draws: int, format_retries: int):
        if variant not in {"untrained", "score_only", "joint_rubric", "two_stage"}:
            raise ValueError(f"Unknown verifier variant: {variant}")
        if draws < 1 or format_retries < 0:
            raise ValueError("draws must be positive and format_retries nonnegative")
        self.model, self.model_name, self.variant = model, model_name, variant
        self.draws, self.format_retries = draws, format_retries

    def score(self, task: str, history: list[Message], candidate: Candidate, role: Role, session: ModelSession) -> Score:
        if not task.strip():
            raise ValueError("Verifier task must be nonempty")
        values: list[float | None] = []
        outputs: list[str] = []
        rubrics: list[str] = []
        for _ in range(self.draws):
            if self.variant == "two_stage":
                generator = [
                    *_prefix(task, history, candidate, load_prompt("two_stage/generator_system.txt").strip()),
                    {"role": "user", "content": load_prompt("two_stage/generator.txt").strip()},
                ]
                current = generator
                rubric = ""
                for attempt in range(self.format_retries + 1):
                    text = _text(session.query(_two_stage_model(self.model, attempt), self.model_name, current, role))
                    outputs.append(text)
                    parsed = parse_generator_response(text)
                    if parsed["format_valid"]:
                        rubric = cast(str, parsed["checklist_text"])
                        break
                    if attempt == self.format_retries:
                        raise InvalidOutput(f"Invalid candidate-specific rubric: {parsed['errors']}")
                    current = [*generator, {"role": "assistant", "content": text},
                               {"role": "user", "content": load_prompt("two_stage/generator_retry.txt").strip()}]
                rubrics.append(rubric)
                request = _render("two_stage/scorer.jinja", {"checklist_text": rubric})
                messages = [
                    *_prefix(task, history, candidate, load_prompt("two_stage/scorer_system.txt").strip()),
                    {"role": "user", "content": request},
                ]
                retry_path = "two_stage/scorer_retry.txt"
            else:
                variables = {
                    "task": task, "candidate": candidate_variables(candidate, 0),
                    "history_message_format": "multi_turn_chat", "history_steps": -1,
                    "all_steps": [], "steps": [], "enable_verbal_feedback": True,
                    "verbal_feedback_mode": "reasoning",
                }
                request = (load_prompt("joint_rubric/scoring.txt") if self.variant == "joint_rubric"
                           else _render(f"{self.variant}/scoring.jinja", variables))
                if self.variant == "joint_rubric":
                    messages = [
                        *_prefix(task, history, candidate, load_prompt("joint_rubric/system.txt")),
                        {"role": "user", "content": request},
                    ]
                else:
                    # Scalar research prompts contain task and proposed action
                    # in the last request, without an extra candidate chat turn.
                    messages = [
                        {"role": "system", "content": _render(f"{self.variant}/system.txt", variables)},
                        *observed_history(history), {"role": "user", "content": request},
                    ]
                retry_path = f"{self.variant}/retry.txt"
                rubric = ""
            current = messages
            if self.variant in {"untrained", "score_only"}:
                # RewardModelVerifier does not retry missing scalar parses.
                # It retains None draws and averages the available scores.
                response = session.query(self.model, self.model_name, messages, role)
                text = str(response.get("content", ""))
                outputs.append(text)
                value = parse_reward(text, REWARD_REGEX)
                values.append(value)
                if value is None:
                    session.diagnostics.append({"event": "missing_scalar_draw", "role": role,
                                                "candidate_id": candidate.id, "draw": len(values)})
                continue
            for attempt in range(self.format_retries + 1):
                model = _two_stage_model(self.model, attempt) if self.variant == "two_stage" else self.model
                text = _text(session.query(model, self.model_name, current, role))
                outputs.append(text)
                try:
                    value = terminal_score(text)
                    if self.variant == "two_stage":
                        parsed_score = parse_scorer_response(text, rubric)
                        if not parsed_score["format_valid"]:
                            raise InvalidOutput(f"Invalid rubric scoring format: {parsed_score['errors']}")
                except InvalidOutput:
                    if attempt == self.format_retries:
                        raise
                    retry = load_prompt(retry_path)
                    if self.variant == "two_stage":
                        retry = retry.strip()
                    current = [*messages, {"role": "assistant", "content": text},
                               {"role": "user", "content": retry}]
                else:
                    values.append(value)
                    break
        available = [value for value in values if value is not None]
        return Score(sum(available) / len(available) if available else None,
                     tuple(outputs), tuple(rubrics), tuple(values))


class SlateSelector:
    """Policy-model selector using the paper's multi-action FINAL-index contract."""

    def __init__(self, model: QueryModel, model_name: str, format_retries: int):
        if format_retries < 0:
            raise ValueError("format_retries must be nonnegative")
        self.model, self.model_name, self.format_retries = model, model_name, format_retries

    def select(self, task: str, history: list[Message], candidates: tuple[Candidate, ...], session: ModelSession) -> tuple[Candidate, tuple[str, ...]]:
        if not candidates:
            raise ValueError("Selector needs at least one valid candidate")
        request = _render("selection/scoring.jinja", {
            "task": task, "candidates": [candidate_variables(candidate, index) for index, candidate in enumerate(candidates)],
            "selection_index_base": 1, "history_message_format": "multi_turn_chat", "history_steps": -1,
            "all_steps": [], "steps": [],
        })
        messages: list[Message] = [
            {"role": "system", "content": load_prompt("selection/system.txt")},
            *observed_history(history),
            {"role": "user", "content": request},
        ]
        outputs: list[str] = []
        current = messages
        for attempt in range(self.format_retries + 1):
            response = session.query(self.model, self.model_name, current, "selector")
            text = str(response.get("content", ""))
            outputs.append(text)
            selected, raw, source = parse_selection(text, len(candidates), SELECTION_REGEX, 1)
            session.diagnostics.append({"event": "selection_parse", "attempt": attempt + 1,
                                        "raw_index": raw, "parsed_index": selected,
                                        "selection_parse_source": source, "malformed": selected is None})
            if selected is not None:
                return candidates[selected], tuple(outputs)
            if attempt == self.format_retries:
                raise InvalidOutput("Selector did not identify an available candidate using the research index parser")
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"^(?:[ \t]*\r?\n)+", "", cleaned)
            current = list(messages)
            if text.strip():
                current.append({"role": "assistant", "content": cleaned})
            current.append({"role": "user", "content": load_prompt("selection/retry.txt").format(
                first_index=1, last_index=len(candidates))})
        raise AssertionError("Selector retries did not return or raise")
