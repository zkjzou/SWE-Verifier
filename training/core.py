"""Deterministic update windows and distributed token-normalized objectives."""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext

from training.sampling import epoch_aware_length_indices


class UpdateStream:
    def __init__(self, records, batch_size):
        self.records, self.batch_size = records, batch_size
        self.windows = math.ceil(len(records) / batch_size)
        self.cycle, self.order = None, None

    def batch(self, update):
        cycle, window = divmod(update, self.windows)
        if cycle != self.cycle:
            self.order = epoch_aware_length_indices(
                self.records,
                epoch_row_ids=[r["row_id"] for r in self.records],
                per_device_batch_size=2,
                world_size=4,
                gradient_accumulation_steps=self.batch_size // 8,
                seed=0,
                epoch=cycle,
            )
            self.order.extend(self.order[i % len(self.records)] for i in range((-len(self.order)) % self.batch_size))
            self.cycle = cycle
        return self.order[window * self.batch_size : (window + 1) * self.batch_size]


def rank_batches(streams, update, rank):
    if not 0 <= rank < 4 or len(streams) not in (1, 2):
        raise ValueError("Expected four ranks and one or two task streams")
    if len(streams) == 2:
        return [(mode, stream.batch(update)[rank * 2 : rank * 2 + 2]) for mode, stream in streams.items()]
    mode, stream = next(iter(streams.items()))
    batch = stream.batch(update)
    return [(mode, batch[micro * 8 + rank * 2 : micro * 8 + rank * 2 + 2]) for micro in range(2)]


def learning_rate(config, completed_updates):
    warmup, maximum, rate = config["warmup_steps"], config["max_steps"], config["learning_rate"]
    if completed_updates < warmup:
        return rate * completed_updates / warmup
    return rate * 0.5 * (1 + math.cos(math.pi * (completed_updates - warmup) / (maximum - warmup)))


def backward_update(model, batches, *, device, autocast=False, after_backward=None):
    """One global update. Each batch is (task, tensors); DDP averages gradients once.

    A single task uses its total supervised tokens across both microsteps and all
    ranks. Shared training averages two independent task means. No extra division
    by the accumulation count is applied.
    """
    import torch
    import torch.distributed as dist

    world = dist.get_world_size() if dist.is_initialized() else 1
    counts = defaultdict(int)
    for mode, batch in batches:
        counts[mode] += int(batch["labels"][..., 1:].ne(-100).sum())
    modes = sorted(counts)
    denominators = torch.tensor([counts[mode] for mode in modes], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(denominators)
    if not bool((denominators > 0).all()):
        raise ValueError("Empty supervised-token denominator")
    denominators = dict(zip(modes, denominators))
    total = torch.zeros((), device=device)
    for index, (mode, batch) in enumerate(batches):
        sync = model.no_sync() if world > 1 and index < len(batches) - 1 else nullcontext()
        with sync:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch, num_items_in_batch=denominators[mode], skip_logits=True)
                if outputs.logits is not None or not bool(torch.isfinite(outputs.loss)):
                    raise ValueError("Expected finite fused loss without materialized logits")
                loss = outputs.loss * world / len(modes)
            loss.backward()
        total += loss.detach()
        del outputs, loss, batch
        if after_backward:
            after_backward()
    if dist.is_initialized():
        dist.all_reduce(total)
        total /= world
    return float(total)
