"""Compare real four-rank DDP gradients with a single-process objective."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from training.core import backward_update
from training.tests.test_training import ToyModel


def batches_for_rank(rank, shared):
    batches = []
    for micro in range(2):
        ids = (torch.arange(14).reshape(2, 7) + rank + micro) % 17
        labels = ids.clone()
        for row in range(2):
            labels[row, : 1 + (rank + micro * 2 + row) % 5] = -100
        batches.append(
            (
                f"task{micro}" if shared else "task",
                {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels},
            )
        )
    return batches


def worker(rank, rendezvous, shared):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(6)
        model = ToyModel()
        reference = ToyModel()
        reference.load_state_dict(model.state_dict())
        ddp = DistributedDataParallel(model)
        observed = backward_update(ddp, batches_for_rank(rank, shared), device=torch.device("cpu"))
        all_batches = sum([batches_for_rank(r, shared) for r in range(4)], [])
        modes = sorted({mode for mode, _ in all_batches})
        loss = 0
        for mode in modes:
            batches = [batch for task, batch in all_batches if task == mode]
            denominator = sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in batches)
            for batch in batches:
                loss = loss + reference(**batch, num_items_in_batch=denominator, skip_logits=True).loss / len(modes)
        loss.backward()
        assert observed == pytest.approx(float(loss.detach()), rel=1e-6)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("shared", [False, True], ids=["single_task", "equal_task_means"])
def test_four_rank_gradients(tmp_path, shared):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(), shared), nprocs=4, join=True)
