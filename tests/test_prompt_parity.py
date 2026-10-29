"""Golden messages recovered independently from saved templates and checkpoint clients."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_verifier.inference.checklist_contract import normalize_qwen35_chat_messages
from swe_verifier.inference.scoring import Scorer
from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import Budget, Candidate, Message

_FIXTURES = json.loads((Path(__file__).parent / "fixtures/prompt_parity.json").read_text())["fixtures"]


class CaptureModel:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.messages: list[list[Message]] = []

    def query(self, messages: list[Message]) -> Message:
        self.messages.append(messages)
        return {"role": "assistant", "content": next(self.outputs)}


@pytest.mark.parametrize("fixture", _FIXTURES, ids=[item["variant"] for item in _FIXTURES])
def test_source_prompt_message_parity(fixture: dict) -> None:
    model = CaptureModel(fixture["model_outputs"])
    scorer = Scorer(model, fixture["model"], fixture["variant"], 1, 0)
    score = scorer.score(
        fixture["task"], fixture["history"], Candidate(**fixture["candidate"]), "gate",
        ModelSession(Budget(10, None)),
    )
    assert score.value == 0.8
    actual = [normalize_qwen35_chat_messages(messages) for messages in model.messages]
    expected = [normalize_qwen35_chat_messages(messages) for messages in fixture["expected_messages"]]
    assert actual == expected
