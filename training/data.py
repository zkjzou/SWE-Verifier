"""Rebuild native IDs and final-response masks from the released message exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from training.messages import normalize_qwen35_chat_messages

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET = "zkjzou/swe-verifier-training"
DATASET_REVISION = "867f17b0004494b32b68d8bb03640073fdd4308e"
MODES = ("score_only", "joint_rubric", "two_stage_generator", "two_stage_scorer")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_example(tokenizer, messages, manifest, max_length=65536, *, target_count_includes_suffix=True):
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected a final assistant supervision target")
    messages = normalize_qwen35_chat_messages(messages)

    def render(value, generation):
        ids = tokenizer.apply_chat_template(
            value, tokenize=True, add_generation_prompt=generation, enable_thinking=False
        )
        return list(ids["input_ids"]) if isinstance(ids, Mapping) else list(ids)

    ids, prompt = render(messages, False), render(messages[:-1], True)
    target = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError("Released row exceeds cutoff; do not silently truncate historical exports")
    if ids[: len(prompt)] != prompt or ids[len(prompt) : len(prompt) + len(target)] != target:
        raise ValueError("Native generation prefix or supervised target differs")
    empty_think = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    if prompt[-len(empty_think) :] != empty_think:
        raise ValueError("Expected masked empty-thinking generation prefix")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") not in ids[len(prompt) + len(target) :]:
        raise ValueError("Missing supervised end-of-turn token")
    labels = [-100] * len(prompt) + ids[len(prompt) :]
    observed = {
        "training_tokens": len(ids),
        "target_tokens": len(ids) - len(prompt),
        "prompt_tokens": len(prompt),
        "input_ids_sha256": digest(ids),
        "labels_sha256": digest(labels),
        "target_sha256": hashlib.sha256(messages[-1]["content"].encode()).hexdigest(),
    }
    # The older Score Only manifest counts target text alone; its trained labels
    # still include the native end-of-turn suffix, as in the rubric variants.
    if not target_count_includes_suffix:
        observed["target_tokens"] = len(target)
    for key, value in observed.items():
        if key in manifest and manifest[key] != value:
            raise ValueError(f"Historical {key} mismatch for row {manifest.get('row_index')}")
    return ids, len(prompt)


class TokenDataset:
    """Memory-map token IDs; construct labels only for the current microbatch."""

    def __init__(self, directory):
        import numpy as np

        self.directory = Path(directory)
        self.receipt = json.loads((self.directory / "manifest.json").read_text())
        self.records = [json.loads(line) for line in (self.directory / "records.jsonl").read_text().splitlines()]
        self.tokens = np.memmap(self.directory / "input_ids.bin", mode="r", dtype="<i4")
        if len(self.records) != self.receipt["rows"] or sum(r["training_tokens"] for r in self.records) != len(
            self.tokens
        ):
            raise ValueError("Incomplete prepared token cache")

    def __getitem__(self, index):
        row = self.records[index]
        ids = self.tokens[row["offset"] : row["offset"] + row["training_tokens"]].astype("int64").tolist()
        return {"input_ids": ids, "labels": [-100] * row["prompt_tokens"] + ids[row["prompt_tokens"] :]}

    def verify(self):
        for name, expected in self.receipt["files"].items():
            if file_hash(self.directory / name) != expected:
                raise ValueError(f"Prepared cache checksum mismatch: {name}")


def collate(examples, pad_token_id):
    import torch

    length = max(len(row["input_ids"]) for row in examples)
    result = {
        key: torch.full((len(examples), length), fill, dtype=torch.long)
        for key, fill in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))
    }
    for index, row in enumerate(examples):
        n = len(row["input_ids"])
        result["input_ids"][index, :n] = torch.tensor(row["input_ids"])
        result["labels"][index, :n] = torch.tensor(row["labels"])
        result["attention_mask"][index, :n] = 1
    return result
