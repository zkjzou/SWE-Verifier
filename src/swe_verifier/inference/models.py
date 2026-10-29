"""Text-only model connectors that do not invoke Mini's policy action parser."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping

import litellm

from swe_verifier.inference.messages import native_message
from swe_verifier.inference.types import Message

logger = logging.getLogger(__name__)


class LiteLLMTextModel:
    """Use Mini's LiteLLM dependency for verifier and teacher completions.

    Model identity, serving endpoint, sampling parameters, timeout, and optional
    token prices are explicit model_kwargs supplied by a release recipe.
    """

    def __init__(self, model_name: str, model_kwargs: Mapping[str, object], retries: int):
        if retries < 0:
            raise ValueError("retries must be nonnegative")
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs)
        self.retries = retries

    def query(self, messages: list[Message]) -> Message:
        return self._query(messages, self.model_kwargs)

    def query_with_output_limit(self, messages: list[Message], max_tokens: int) -> Message:
        return self._query(messages, {**self.model_kwargs, "max_tokens": max_tokens})

    def _query(self, messages: list[Message], options: dict[str, object]) -> Message:
        for attempt in range(self.retries + 1):
            try:
                result = litellm.completion(
                    model=self.model_name,
                    messages=[native_message(message) for message in messages],
                    **options,
                )
                break
            except (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError, litellm.InternalServerError) as exc:
                if attempt == self.retries:
                    raise
                logger.warning("Model transport retry", extra={
                    "model": self.model_name, "attempt": attempt + 1, "error": str(exc),
                })
                time.sleep(min(2 ** attempt, 8))
        text = result.choices[0].message.content
        if not isinstance(text, str):
            raise ValueError(f"Model {self.model_name} returned no text completion")
        cost = result._hidden_params.get("response_cost")
        if cost is None:
            try:
                cost = litellm.cost_calculator.completion_cost(result, model=self.model_name)
            except (litellm.NotFoundError, ValueError) as exc:
                logger.warning("Model price unavailable", extra={"model": self.model_name, "error": str(exc)})
        return {
            "role": "assistant", "content": text,
            "extra": {"usage": result.usage.model_dump(), "cost": cost, "response": result.model_dump()},
        }
