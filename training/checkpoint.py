"""Atomic full-state checkpoints and rank-local random-state restoration."""

from __future__ import annotations

import json
import random
from pathlib import Path

from training.core import learning_rate


def topology():
    import torch.distributed as dist

    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def barrier():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def read_checkpoint(path):
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text())
    if state["schema_version"] != 1 or not (path / "optimizer.pt").is_file():
        raise ValueError("Incomplete training checkpoint")
    for rank in range(state["world_size"]):
        if not (path / f"rng-{rank}.pt").is_file():
            raise ValueError("Missing rank RNG state")
    return state


def save_checkpoint(output, step, model, tokenizer, optimizer, binding, config, smoke):
    import torch

    rank, world = topology()
    final = Path(output) / f"checkpoint-{step}"
    staging = Path(output) / f".checkpoint-{step}.incomplete"
    if rank == 0:
        if final.exists():
            raise FileExistsError(final)
        staging.mkdir(exist_ok=False)
    barrier()
    rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(rng, staging / f"rng-{rank}.pt")
    if rank == 0:
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        if tokenizer is not None:
            tokenizer.save_pretrained(staging)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    barrier()
    if rank == 0:
        state = {
            "schema_version": 1,
            "global_step": step,
            "world_size": world,
            "binding": binding,
            "recipe": config,
            "smoke": smoke,
            "scheduler": {"completed_updates": step, "next_learning_rate": learning_rate(config, step)},
        }
        (staging / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
        staging.rename(final)
    barrier()
    return final


def restore_optimizer_rng(path, optimizer, binding):
    import torch

    rank, world = topology()
    state = read_checkpoint(path)
    if state["binding"] != binding or state["world_size"] != world:
        raise ValueError("Resume recipe, data, initialization, or world size differs")
    if state["scheduler"]["next_learning_rate"] != learning_rate(state["recipe"], state["global_step"]):
        raise ValueError("Checkpoint scheduler disagrees with update cursor")
    optimizer.load_state_dict(torch.load(Path(path) / "optimizer.pt", map_location="cpu", weights_only=True))
    rng = torch.load(Path(path) / f"rng-{rank}.pt", map_location="cpu", weights_only=True)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"])
    return state["global_step"]
