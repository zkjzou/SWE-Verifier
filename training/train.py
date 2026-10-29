"""Train the paper variants with four H200 GPUs using native-token SFT."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import timedelta
from pathlib import Path

from training.checkpoint import barrier, read_checkpoint, restore_optimizer_rng, save_checkpoint
from training.core import UpdateStream, backward_update, learning_rate, rank_batches
from training.data import TokenDataset, collate, digest, file_hash

RECIPES = ("score_only", "joint_rubric", "two_stage_scorer", "two_stage_shared")


def load_recipe(name):
    if name not in RECIPES:
        raise ValueError("Unknown training recipe")
    return json.loads(Path(__file__).with_name("configs").joinpath(name + ".json").read_text())


def initialization(config, parent, smoke):
    if "parent_recipe" not in config:
        if parent is not None:
            raise ValueError("Only shared Two-Stage training accepts --init-from")
        return {"kind": "base", "model": config["base_model"], "revision": config["base_revision"]}
    if parent is None:
        raise ValueError("Shared training requires --init-from pointing to the trained scorer checkpoint")
    state = read_checkpoint(parent)
    if state["recipe"] != load_recipe(config["parent_recipe"]) or state["smoke"] != smoke:
        raise ValueError("Wrong scorer parent recipe/scope")
    if not smoke and state["global_step"] != config["parent_step"]:
        raise ValueError("Production shared training must initialize from scorer step 5703")
    # Bind actual weight files so a modified parent cannot silently change initialization.
    weights = sorted(Path(parent).glob("*.safetensors"))
    if not weights:
        raise ValueError("Parent checkpoint has no model weights")
    return {
        "kind": "scorer_weights_fresh_optimizer",
        "parent_state": digest(state),
        "weights": {p.name: file_hash(p) for p in weights},
    }


def run(args):
    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoTokenizer

    from training.model import load_model, make_optimizer, release_host_cache

    config = load_recipe(args.recipe)
    if int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available():
        raise ValueError("Launch with torchrun --nproc-per-node=4 on a four-H200 node")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_properties(device).total_memory < 120 * 1024**3:
        raise ValueError("These 65,536-token recipes target H200-class GPU memory (at least 120 GiB per GPU)")
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        datasets = {mode: TokenDataset(args.data / mode) for mode in config["streams"]}
        for mode, data in datasets.items():
            receipt = data.receipt
            if receipt["mode"] != mode or receipt["scope"] != ("smoke" if args.smoke else "production"):
                raise ValueError("Data mode or preparation scope mismatch")
            for key in ("base_revision", "dataset_revision", "max_length"):
                if receipt[key] != config[key]:
                    raise ValueError(f"Prepared data {key} differs from recipe")
            if not args.smoke and receipt["rows"] != config["rows_per_stream"]:
                raise ValueError("Training row count differs from paper export")
            if rank == 0:
                data.verify()
        barrier()
        init = initialization(config, args.init_from, args.smoke)
        if args.init_from is not None:
            parent = read_checkpoint(args.init_from)
            if parent["binding"]["data"].get("two_stage_scorer") != digest(datasets["two_stage_scorer"].receipt):
                raise ValueError("Shared training scorer data differs from its initialization checkpoint")
        binding = {
            "recipe": digest(config),
            "data": {k: digest(v.receipt) for k, v in datasets.items()},
            "initialization": init,
            "smoke": args.smoke,
        }
        resumed = read_checkpoint(args.resume) if args.resume else None
        if resumed and resumed["binding"] != binding:
            raise ValueError("Resume binding changed")
        start = resumed["global_step"] if resumed else 0
        stop = args.max_updates if args.max_updates is not None else config["max_steps"]
        if args.smoke and args.max_updates is None:
            raise ValueError("--smoke requires an explicit small --max-updates limit")
        if not start < stop <= config["max_steps"]:
            raise ValueError("--max-updates is an absolute stop cursor within the recipe schedule")
        if rank == 0:
            if args.resume is None and args.output.exists() and any(args.output.iterdir()):
                raise FileExistsError("Fresh training requires an empty output directory")
            if args.resume is not None and args.resume.parent.resolve() != args.output.resolve():
                raise ValueError("Resume checkpoint must belong to this output directory")
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "recipe.json").write_text(json.dumps(config, indent=2) + "\n")
        barrier()
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        source = args.resume or args.init_from
        if source is None:
            # Download once on rank zero; subsequent ranks read the same local cache.
            if rank == 0:
                snapshot_download(config["base_model"], revision=config["base_revision"])
            barrier()
            source = snapshot_download(config["base_model"], revision=config["base_revision"], local_files_only=True)
        model = load_model(source, trained_checkpoint=bool(args.resume or args.init_from)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            next(iter(datasets.values())).directory / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        optimizer = make_optimizer(model, config)
        ddp = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        model.train()
        if args.resume:
            restore_optimizer_rng(args.resume, optimizer, binding)
        streams = {mode: UpdateStream(data.records, 8 if len(datasets) == 2 else 16) for mode, data in datasets.items()}
        for update in range(start, stop):
            batches = [
                (mode, collate([datasets[mode][i] for i in indices], tokenizer.pad_token_id))
                for mode, indices in rank_batches(streams, update, rank)
            ]
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(config, update)
            for group in optimizer.param_groups:
                group["lr"] = rate
            loss = backward_update(ddp, batches, device=device, autocast=True, after_backward=release_host_cache)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            step = update + 1
            if rank == 0 and (step == 1 or step % 10 == 0 or step == stop):
                metric = {
                    "step": step,
                    "loss": loss,
                    "lr": rate,
                    "gradient_norm": float(norm),
                    "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
                }
                print(json.dumps(metric), flush=True)
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metric) + "\n")
            if step in config["save_steps"] or step == stop:
                save_checkpoint(args.output, step, model, tokenizer, optimizer, binding, config, args.smoke)
        if rank == 0:
            print(
                json.dumps({"completed_updates": stop, "paper_selected_step": config["paper_selected_step"]}),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-updates", type=int, help="Stop after this absolute update, retaining the full LR schedule"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(load_recipe(args.recipe), indent=2))
        return
    if args.data is None or args.output is None:
        parser.error("--data and --output are required for training")
    run(args)


if __name__ == "__main__":
    main()
