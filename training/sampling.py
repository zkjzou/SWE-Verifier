"""Historical epoch-aware length ordering; global windows remain size 16 or 8."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

APPROXIMATE_LENGTH_BUCKET_TOKENS = 2048


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _approximate_bucket(length: int) -> int:
    _require(length > 0, "sequence lengths must be positive")
    return (int(length) - 1) // APPROXIMATE_LENGTH_BUCKET_TOKENS


def epoch_aware_length_indices(
    manifests: Sequence[Mapping[str, Any]],
    *,
    epoch_row_ids: Sequence[str],
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """Return the exact epoch subset in deterministic, approximate-length-local windows.

    Rows are first shuffled together without using source, trajectory, slate, or sampler-model identity.  A stable
    sort then groups only by a coarse token-length bucket, preserving the shuffled order inside each bucket.  Full
    optimizer windows are shuffled as units and the single incomplete global window, if any, is always retained last.
    """

    _require(per_device_batch_size > 0, "per-device batch size must be positive")
    _require(world_size > 0, "world size must be positive")
    _require(gradient_accumulation_steps > 0, "gradient accumulation must be positive")
    _require(epoch >= 0, "epoch must be nonnegative")
    optimizer_window = per_device_batch_size * world_size * gradient_accumulation_steps
    by_id: dict[str, int] = {}
    for index, row in enumerate(manifests):
        row_id = str(row.get("row_id") or "")
        _require(row_id != "" and row_id not in by_id, f"invalid or duplicate manifest row ID: {row_id!r}")
        by_id[row_id] = index
    _require(bool(epoch_row_ids), "epoch row journal is empty")
    _require(len(set(epoch_row_ids)) == len(epoch_row_ids), "epoch row journal contains duplicate IDs")
    missing = set(epoch_row_ids) - set(by_id)
    _require(not missing, f"epoch row IDs are absent from the union dataset: {sorted(missing)[:3]}")

    shuffled = [by_id[row_id] for row_id in epoch_row_ids]
    random.Random(seed + epoch).shuffle(shuffled)
    shuffled.sort(key=lambda index: _approximate_bucket(int(manifests[index]["training_tokens"])))
    windows = [shuffled[start : start + optimizer_window] for start in range(0, len(shuffled), optimizer_window)]
    partial = windows.pop() if windows and len(windows[-1]) < optimizer_window else None
    random.Random(seed + epoch + 1_000_003).shuffle(windows)
    if partial is not None:
        windows.append(partial)
    ordered = [index for window in windows for index in window]
    _require(len(ordered) == len(epoch_row_ids), "sampler changed the epoch denominator")
    _require(set(ordered) == {by_id[row_id] for row_id in epoch_row_ids}, "sampler changed epoch membership")
    if len(epoch_row_ids) % optimizer_window:
        _require(len(windows[-1]) == len(epoch_row_ids) % optimizer_window, "partial global window is not last")
    return ordered
