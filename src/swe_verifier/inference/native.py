"""Pinned-tokenizer Qwen3.5 /completions connector used by released checkpoints."""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping

import requests

from swe_verifier.inference.checklist_contract import normalize_qwen35_chat_messages
from swe_verifier.inference.types import Message

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.getLogger(__name__)


class NativeTextModel:
    """Render locally with a pinned tokenizer; never execute checkpoint Python files."""

    def __init__(self, model_name: str, api_base: str, tokenizer_path: str, tokenizer_revision: str,
                 model_kwargs: Mapping[str, object], retries: int, timeout_seconds: float):
        if AutoTokenizer is None:
            raise ImportError("Native inference requires installation with the 'native' extra")
        if retries < 0 or timeout_seconds <= 0:
            raise ValueError("retries must be nonnegative and timeout_seconds positive")
        if not tokenizer_revision:
            raise ValueError("Native tokenizer requires an explicit immutable revision")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=tokenizer_revision, trust_remote_code=False)
        self.model_name, self.api_base = model_name, api_base.rstrip("/")
        self.model_kwargs, self.retries, self.timeout_seconds = dict(model_kwargs), retries, timeout_seconds

    def query(self, messages: list[Message]) -> Message:
        return self._query(messages, self.model_kwargs)

    def query_with_output_limit(self, messages: list[Message], max_tokens: int) -> Message:
        return self._query(messages, {**self.model_kwargs, "max_tokens": max_tokens})

    def _query(self, messages: list[Message], model_options: Mapping[str, object]) -> Message:
        ids = self.tokenizer.apply_chat_template(
            normalize_qwen35_chat_messages(messages), tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        empty_think = self.tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        if not empty_think or ids[-len(empty_think):] != empty_think:
            raise ValueError("Pinned tokenizer does not produce the Qwen3.5 empty-thinking prefix")
        options = dict(model_options)
        api_key = options.pop("api_key", None)
        context_limit = int(options.pop("context_limit", 262144))
        input_price = options.pop("input_cost_per_token", None)
        output_price = options.pop("output_cost_per_token", None)
        max_tokens = options.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Native model requires a positive explicit max_tokens")
        if len(ids) + max_tokens > context_limit:
            raise ValueError("Native prompt plus output allowance exceeds context limit; no automatic truncation")
        stop = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if stop != 248046:
            raise ValueError("Pinned tokenizer has an unexpected Qwen3.5 termination token")
        prohibited = {"model", "prompt", "n", "stream", "stop_token_ids", "ignore_eos", "add_special_tokens"} & options.keys()
        if prohibited:
            raise ValueError(f"Native model options cannot override protocol fields: {sorted(prohibited)}")
        payload = {**options, "model": self.model_name.removeprefix("openai/"), "prompt": ids, "n": 1,
                   "stream": False, "stop_token_ids": [stop], "ignore_eos": True, "add_special_tokens": False}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        endpoint = self.api_base + "/completions"
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout_seconds)
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
                if attempt == self.retries or (status is not None and status < 500 and status != 429):
                    body = exc.response.text if isinstance(exc, requests.HTTPError) and exc.response is not None else str(exc)
                    raise RuntimeError(f"Native request failed at {endpoint}, model={self.model_name}, status={status}: {body}") from exc
                logger.warning("Native transport retry", extra={"endpoint": endpoint, "attempt": attempt + 1, "status": status})
                time.sleep(min(2 ** attempt, 8))
        result = response.json()
        usage = result.get("usage", {})
        if result.get("model") != payload["model"] or usage.get("prompt_tokens") != len(ids):
            raise ValueError("Native server model identity or prompt-token count differs from the pinned request")
        choices = result.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0].get("text"), str):
            raise ValueError("Native server must return exactly one text completion")
        if choices[0].get("finish_reason") not in {"stop", "length"}:
            raise ValueError(f"Native generation aborted: {choices[0].get('finish_reason')}")
        cost = None
        if input_price is not None or output_price is not None:
            if not all(isinstance(p, (int, float)) and math.isfinite(p) and p >= 0 for p in (input_price, output_price)):
                raise ValueError("Both token prices must be finite and nonnegative when configured")
            cost = input_price * usage["prompt_tokens"] + output_price * usage["completion_tokens"]
        return {"role": "assistant", "content": choices[0]["text"],
                "extra": {"usage": usage, "cost": cost, "response": result}}
