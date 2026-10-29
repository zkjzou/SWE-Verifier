"""Download pinned training exports and verify native tokenization before training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

from training.data import BASE_MODEL, BASE_REVISION, DATASET, DATASET_REVISION, MODES, encode_example, file_hash


def prepare_mode(source, output, mode, tokenizer, contract, limit):
    import numpy as np

    final = output / mode
    if final.exists():
        raise FileExistsError(f"Prepared data already exists: {final}")
    staging = output / (mode + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    artifacts = contract["datasets"][mode]["artifacts"]
    for name, entry in artifacts.items():
        if file_hash(source / entry["file"]) != entry["gzip_sha256"]:
            raise ValueError(f"Compressed source hash mismatch: {name}")
    hashes = [hashlib.sha256(), hashlib.sha256()]
    count, offset = 0, 0
    with (
        gzip.open(source / artifacts["train.jsonl"]["file"], "rb") as rows,
        gzip.open(source / artifacts["train_manifest.jsonl"]["file"], "rb") as manifests,
        (staging / "input_ids.bin").open("wb") as tokens,
        (staging / "records.jsonl").open("w") as records,
    ):
        pairs = zip(rows, manifests, strict=True)
        if limit is not None:
            pairs = itertools.islice(pairs, limit)
        for row_line, meta_line in pairs:
            hashes[0].update(row_line)
            hashes[1].update(meta_line)
            row, meta = json.loads(row_line), json.loads(meta_line)
            if meta["row_index"] != count:
                raise ValueError("Non-contiguous original row order")
            ids, prompt = encode_example(
                tokenizer, row["messages"], meta, target_count_includes_suffix=mode != "score_only"
            )
            np.asarray(ids, dtype="<i4").tofile(tokens)
            records.write(
                json.dumps(
                    {"row_id": meta["row_id"], "offset": offset, "training_tokens": len(ids), "prompt_tokens": prompt}
                )
                + "\n"
            )
            offset += len(ids)
            count += 1
            if count % 1000 == 0:
                print(f"{mode}: {count} verified rows", flush=True)
    if limit is None:
        for h, name in zip(hashes, ("train.jsonl", "train_manifest.jsonl")):
            if h.hexdigest() != artifacts[name]["original_sha256"] or count != artifacts[name]["rows"]:
                raise ValueError("Decompressed historical export hash/count mismatch")
    if count == 0:
        raise ValueError("Empty preparation")
    tokenizer.save_pretrained(staging / "tokenizer")
    receipt = {
        "mode": mode,
        "rows": count,
        "scope": "smoke" if limit is not None else "production",
        "dataset_repo": DATASET,
        "dataset_revision": DATASET_REVISION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "max_length": 65536,
        "source_artifacts": artifacts,
        "files": {str(p.relative_to(staging)): file_hash(p) for p in staging.rglob("*") if p.is_file()},
    }
    (staging / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    staging.rename(final)
    print(f"{mode}: prepared {count} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--dataset-dir", type=Path, help="Existing exact Hub snapshot; avoids downloading exports")
    parser.add_argument("--tokenizer", type=Path, help="Local pinned base tokenizer; token hashes are still checked")
    parser.add_argument("--limit", type=int, help="Prepare smoke-only data; production training rejects it")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    source = args.dataset_dir or Path(
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["release-manifest.json", *[f"{mode}/train*.jsonl.gz" for mode in args.modes]],
        )
    )
    tokenizer_path = args.tokenizer or Path(
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            allow_patterns=["tokenizer*", "chat_template*", "vocab.json", "merges.txt", "config.json"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    contract = json.loads((source / "release-manifest.json").read_text())
    pinned = json.loads(Path(__file__).with_name("artifacts.json").read_text())["configurations"]
    for mode in args.modes:
        if contract["datasets"][mode]["artifacts"] != pinned[mode]["artifacts"]:
            raise ValueError(f"Source artifacts differ from pinned release: {mode}")
    # Verify the source tokenizer files, including the native template, even for Score Only rows without token hashes.
    expected = json.loads(Path(__file__).with_name("tokenizer-hashes.json").read_text())
    for name, checksum in expected.items():
        if file_hash(tokenizer_path / name) != checksum:
            raise ValueError(f"Pinned tokenizer artifact differs: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        prepare_mode(source, args.output, mode, tokenizer, contract, args.limit)


if __name__ == "__main__":
    main()
