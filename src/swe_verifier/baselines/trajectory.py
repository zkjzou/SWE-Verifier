"""Pairwise trajectory tournament and a strictly offline Oracle@k upper bound."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import combinations

from jinja2 import StrictUndefined, Template

from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import InvalidOutput, QueryModel, Usage
from swe_verifier.prompts import load_prompt


@dataclass(frozen=True)
class Trajectory:
    id: str
    trace: str
    patch: str


@dataclass(frozen=True)
class Comparison:
    left_id: str
    right_id: str
    criterion: str
    repetition: int
    left_score: float
    right_score: float
    output: str
    scoring_mode: str


@dataclass(frozen=True)
class TrajectoryDecision:
    selected: Trajectory
    wins: tuple[float, ...]
    comparisons: tuple[Comparison, ...]
    usage: tuple[Usage, ...]


def _letter_score(text: str, tag: str) -> float:
    match = re.search(rf"<{tag}>\s*([A-Ta-t])\s*</{tag}>", text)
    if match is None:
        raise InvalidOutput(f"Trajectory judge must return one A-T letter in <{tag}> tags")
    return (19 - (ord(match[1].upper()) - ord("A"))) / 19


def _expected_score(response: dict[str, object], tag: str) -> float | None:
    """Use top-token log probabilities when the provider returns them."""
    extra = response.get("extra", {})
    if not isinstance(extra, dict):
        return None
    raw = extra.get("response", {})
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices", [])
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict) or not isinstance(logprobs.get("content"), list):
        return None
    tokens = logprobs["content"]
    prefix = ""
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            return None
        prefix += str(token.get("token", ""))
        if prefix.rstrip().endswith(f"<{tag}>") and index + 1 < len(tokens):
            next_token = tokens[index + 1]
            if not isinstance(next_token, dict):
                return None
            alternatives = next_token.get("top_logprobs", [])
            weights: dict[int, float] = {}
            for alternative in alternatives:
                if not isinstance(alternative, dict):
                    continue
                letter = str(alternative.get("token", "")).strip().upper()
                logprob = alternative.get("logprob")
                if len(letter) == 1 and "A" <= letter <= "T" and isinstance(logprob, (float, int)) and math.isfinite(logprob):
                    value = 19 - (ord(letter) - ord("A"))
                    weights[value] = max(weights.get(value, 0.0), math.exp(logprob))
            total = sum(weights.values())
            return sum(value * weight for value, weight in weights.items()) / total / 19 if total else None
    return None


def select_trajectories(task: str, trajectories: tuple[Trajectory, ...], model: QueryModel,
                        model_name: str, repetitions: int, session: ModelSession) -> TrajectoryDecision:
    """Rank all completed rollouts without receiving test outcomes or gold patches.

    Ties use input order, matching the original tournament. Every pair is judged;
    benchmark outcome labels never determine which pairs reach the verifier.
    """
    if len(trajectories) < 2 or len({item.id for item in trajectories}) != len(trajectories):
        raise ValueError("Trajectory comparison needs at least two rollouts with unique IDs")
    if repetitions < 1:
        raise ValueError("Trajectory repetitions must be positive")
    config = json.loads(load_prompt("trajectory/criteria.json"))
    template = Template(load_prompt("trajectory/pairwise.jinja"), undefined=StrictUndefined)
    start = len(session.usage)
    wins = [0.0] * len(trajectories)
    comparisons: list[Comparison] = []
    for left, right in combinations(range(len(trajectories)), 2):
        left_values: list[float] = []
        right_values: list[float] = []
        for criterion in config["criteria"]:
            prompt = template.render(
                problem=task, trace_a=trajectories[left].trace + "\n\n" + trajectories[left].patch,
                trace_b=trajectories[right].trace + "\n\n" + trajectories[right].patch,
                criterion=criterion, ground_truth_note=config["ground_truth_note"],
            )
            for repetition in range(repetitions):
                response = session.query(model, model_name, [{"role": "user", "content": prompt}], "trajectory")
                output = str(response.get("content", ""))
                a, b = _expected_score(response, "score_A"), _expected_score(response, "score_B")
                mode = "logprob_expectation"
                if a is None or b is None:
                    a, b = _letter_score(output, "score_A"), _letter_score(output, "score_B")
                    mode = "parsed_letters"
                left_values.append(a)
                right_values.append(b)
                comparisons.append(Comparison(trajectories[left].id, trajectories[right].id, criterion["id"],
                                              repetition, a, b, output, mode))
        a_mean, b_mean = sum(left_values) / len(left_values), sum(right_values) / len(right_values)
        wins[left] += 1.0 if a_mean > b_mean else 0.5 if a_mean == b_mean else 0.0
        wins[right] += 1.0 if b_mean > a_mean else 0.5 if a_mean == b_mean else 0.0
    selected = max(range(len(wins)), key=lambda index: wins[index])
    return TrajectoryDecision(trajectories[selected], tuple(wins), tuple(comparisons), tuple(session.usage[start:]))


def oracle_at_k(outcomes: tuple[bool, ...]) -> bool:
    """Offline upper bound only: whether any evaluated rollout solved the task."""
    if not outcomes or any(type(outcome) is not bool for outcome in outcomes):
        raise ValueError("Oracle requires a nonempty sequence of boolean benchmark outcomes")
    return any(outcomes)
