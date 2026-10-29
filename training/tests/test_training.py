"""CPU tests for native masks, data integrity, ordering, and checkpoint continuation."""

import copy
import json
import random
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from training.checkpoint import read_checkpoint, restore_optimizer_rng, save_checkpoint
from training.core import UpdateStream, learning_rate, rank_batches
from training.data import collate, digest, encode_example
from training.messages import normalize_qwen35_chat_messages
from training.train import initialization, load_recipe


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and not enable_thinking
        return UserDict(input_ids=[10, 11, 12] if add_generation_prompt else [10, 11, 12, 13, 14])

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [11, 12] if text == "<think>\n\n</think>\n\n" else [13]

    def convert_tokens_to_ids(self, token):
        return 14


def test_native_masks_mapping_api_and_manifest():
    messages = [{"role": "user", "content": "input"}, {"role": "assistant", "content": "answer"}]
    manifest = {
        "training_tokens": 5,
        "target_tokens": 2,
        "prompt_tokens": 3,
        "input_ids_sha256": digest([10, 11, 12, 13, 14]),
        "labels_sha256": digest([-100, -100, -100, 13, 14]),
    }
    assert encode_example(FakeTokenizer(), messages, manifest) == ([10, 11, 12, 13, 14], 3)
    assert encode_example(
        FakeTokenizer(), messages, {**manifest, "target_tokens": 1}, target_count_includes_suffix=False
    ) == ([10, 11, 12, 13, 14], 3)
    with pytest.raises(ValueError, match="labels_sha256"):
        encode_example(FakeTokenizer(), messages, {**manifest, "labels_sha256": "changed"})
    with pytest.raises(ValueError, match="cutoff"):
        encode_example(FakeTokenizer(), messages, manifest, max_length=4)


def test_native_tools_are_normalized_without_mutating_exports():
    messages = [{"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]}]
    original = copy.deepcopy(messages)
    result = normalize_qwen35_chat_messages(messages)
    assert result[0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert messages == original
    messages[0]["tool_calls"][0]["function"]["arguments"] = "invalid JSON"
    with pytest.raises(ValueError, match="invalid JSON"):
        normalize_qwen35_chat_messages(messages)


def test_collation_masks_prompt_and_padding():
    batch = collate([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4, 5], "labels": [-100, 5]}], 0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


@pytest.mark.parametrize("size", [8, 16])
def test_stream_membership_prefix_padding_and_resume(size):
    rows = [{"row_id": str(i), "training_tokens": 1 + (i * 379) % 9000} for i in range(35)]
    stream = UpdateStream(rows, size)
    first = sum([stream.batch(i) for i in range(stream.windows)], [])
    assert sorted(first[:35]) == list(range(35))
    assert first[35:] == first[: len(first) - 35]
    assert stream.batch(stream.windows) != first[:size]
    for cursor in [0, 2, stream.windows, stream.windows * 3 + 2]:
        assert stream.batch(cursor) == UpdateStream(rows, size).batch(cursor)


def test_rank_partition_covers_single_and_shared_updates():
    rows = [{"row_id": str(i), "training_tokens": 10} for i in range(32)]
    single = {"one": UpdateStream(rows, 16)}
    shards = [rank_batches(single, 0, rank) for rank in range(4)]
    assert sorted(i for batch in shards for _, ids in batch for i in ids) == sorted(single["one"].batch(0))
    shared = {mode: UpdateStream(rows, 8) for mode in ("scorer", "generator")}
    for mode in shared:
        indices = [i for rank in range(4) for task, ids in rank_batches(shared, 0, rank) if task == mode for i in ids]
        assert indices == shared[mode].batch(0)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(17, 5)
        self.linear = torch.nn.Linear(5, 17)

    def forward(self, input_ids, attention_mask, labels, num_items_in_batch, skip_logits):
        assert skip_logits
        logits = self.linear(self.embed(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1), reduction="sum")
        return SimpleNamespace(loss=loss / num_items_in_batch, logits=None)

    def save_pretrained(self, directory, **kwargs):
        torch.save(self.state_dict(), Path(directory) / "model.pt")


def test_resume_restores_optimizer_rng_and_schedule(tmp_path):
    config = load_recipe("joint_rubric")
    binding = {"data": "immutable"}
    torch.manual_seed(12)
    random.seed(12)
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def update(m, opt, step):
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = learning_rate(config, step)
        loss = sum(p.square().sum() * torch.rand(()) * random.random() for p in m.parameters())
        loss.backward()
        opt.step()

    update(model, optimizer, 0)
    checkpoint = save_checkpoint(tmp_path, 1, model, None, optimizer, binding, config, False)
    update(model, optimizer, 1)
    expected = copy.deepcopy(model.state_dict())
    restored = ToyModel()
    restored.load_state_dict(torch.load(checkpoint / "model.pt", weights_only=True))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    assert restore_optimizer_rng(checkpoint, restored_optimizer, binding) == 1
    update(restored, restored_optimizer, 1)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="differs"):
        restore_optimizer_rng(checkpoint, restored_optimizer, {"data": "changed"})
    (checkpoint / "rng-0.pt").unlink()
    with pytest.raises(ValueError, match="RNG"):
        read_checkpoint(checkpoint)


def test_shared_requires_completed_scorer_and_fresh_optimizer(tmp_path):
    config = load_recipe("two_stage_shared")
    with pytest.raises(ValueError, match="requires --init-from"):
        initialization(config, None, False)
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    parent = save_checkpoint(tmp_path, 5703, model, None, optimizer, {}, load_recipe("two_stage_scorer"), False)
    (parent / "model.safetensors").write_bytes(b"test checkpoint fingerprint")
    init = initialization(config, parent, False)
    assert init["kind"] == "scorer_weights_fresh_optimizer"
    with pytest.raises(ValueError, match="scope"):
        initialization(config, parent, True)
    state_path = parent / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["global_step"] = 3802
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="5703"):
        initialization(config, parent, False)
