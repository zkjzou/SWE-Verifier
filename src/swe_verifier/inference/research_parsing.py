"""Scalar and selection parsing from SWE-PRM's research verifiers.

Source: mini-swe-agent/src/minisweagent/verifiers/{reward_model,llm}.py
at fa12602ab5eae519bd92384f118c91c785641ce9. The released SCORE and
FINAL prompt profiles supply the configured regexes below. Keep source
pattern precedence, last-match behavior, case sensitivity, and index checks.
"""
from __future__ import annotations

import re

REWARD_REGEX = r"SCORE:\s*([+-]?\d+(?:\.\d+)?)"
SELECTION_REGEX = r"FINAL:\s*(\d+)"
_EXPLICIT_SELECTION_PATTERNS = (
    re.compile(
        r"^\s*(?:FINAL|SELECTED|SELECTION|ANSWER|CHOICE)"
        r"(?:\s+(?:ANSWER|SELECTION|CANDIDATE))?\s*[:=-]\s*"
        r"(?:Candidate\s*)?#?\s*(\d+)\b(?!\.\d)", re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:I\s+)?(?:choose|select|pick)\s+(?:Candidate\s*)?#?\s*(\d+)\b(?!\.\d)",
        re.IGNORECASE | re.MULTILINE,
    ),
)


def parse_reward(content: str, configured_pattern: str) -> float | None:
    """Use the configured pattern, then the research FINAL/SCORE/REWARD patterns."""
    patterns = [configured_pattern] if configured_pattern.strip() else []
    for pattern in (r"FINAL:\s*([+-]?\d+(?:\.\d+)?)", r"SCORE:\s*([+-]?\d+(?:\.\d+)?)",
                    r"REWARD:\s*([+-]?\d+(?:\.\d+)?)"):
        if pattern not in patterns:
            patterns.append(pattern)
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        if not matches:
            continue
        raw = matches[-1]
        if isinstance(raw, tuple):
            raw = raw[0]
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def parse_selection(content: str, n_candidates: int, configured_pattern: str,
                    index_base: int) -> tuple[int | None, int | None, str | None]:
    """Return selected position, raw index, and research parse-source metadata."""
    raw_index: int | None = None
    source: str | None = None
    for pattern in _EXPLICIT_SELECTION_PATTERNS:
        matches = list(pattern.finditer(content))
        if matches:
            raw_index, source = int(matches[-1].group(1)), "explicit_final"
            break
    if raw_index is None:
        for raw in reversed(re.findall(configured_pattern, content, re.MULTILINE)):
            if isinstance(raw, tuple):
                raw = next((item for item in raw if item), None)
            try:
                raw_index, source = int(raw), "selection_regex"
                break
            except (ValueError, TypeError):
                continue
    if raw_index is None:
        return None, None, source
    selected = raw_index - index_base
    return (selected if 0 <= selected < n_candidates else None), raw_index, source
