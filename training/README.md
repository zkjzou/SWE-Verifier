# Training on four H200 GPUs

Train Score Only, Joint Rubric, and Two-Stage Rubric from the paper's released message exports. These scripts use one node with **4 × H200 (141 GB)**, full parameter training of the language model, BF16 computation, FP32 trainable weights, and a frozen vision tower/projector. Checkpoint inputs are offloaded to host memory; use a node with ample host RAM and local storage for token caches and full optimizer checkpoints.

Run commands from the repository root. Use a separate Python 3.11 or 3.12 environment from inference, with a CUDA 13.0 toolkit and compatible NVIDIA driver:

```bash
python3 -m venv training/.venv
source training/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r training/requirements.txt
MAX_JOBS=8 python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

Use `hf auth login` if the dataset or base model requires authentication. Credentials stay in the local Hugging Face credential store.

## Prepare the data

```bash
python -m training.prepare --output training/data
```

This downloads the pinned [training dataset](https://huggingface.co/datasets/zkjzou/swe-verifier-training) and Qwen/Qwen3.5-4B tokenizer, verifies source checksums, and creates memory-mapped token caches. It preserves the original message order and history truncation, masks all tokens before the final assistant response, and includes the native end-of-turn suffix in the target. Rows exceeding 65,536 tokens or disagreeing with saved token/mask fingerprints fail preparation. `raw_teacher` is supplementary and is not a student training input.

`--modes score_only` prepares a single configuration. `--dataset-dir PATH --tokenizer PATH` uses local snapshots with the same checksum checks. Preparation refuses to overwrite an existing cache; interrupted preparations leave a `.incomplete` directory that must be removed before retrying.

## Train

```bash
# Score Only
bash training/launch.sh --recipe score_only \
  --data training/data --output training/outputs/score_only

# Joint Rubric
bash training/launch.sh --recipe joint_rubric \
  --data training/data --output training/outputs/joint_rubric

# Two-Stage Rubric: train the scorer, then initialize shared training from it
bash training/launch.sh --recipe two_stage_scorer \
  --data training/data --output training/outputs/two_stage_scorer
bash training/launch.sh --recipe two_stage_shared \
  --data training/data --output training/outputs/two_stage_shared \
  --init-from training/outputs/two_stage_scorer/checkpoint-5703
```

| Recipe | Training rows | Maximum updates | Paper checkpoint |
|---|---:|---:|---:|
| `score_only` | 30,494 | 5,718 | 5,718 |
| `joint_rubric` | 30,403 | 5,703 | 3,802 |
| `two_stage_scorer` | 30,403 | 5,703 | Initialization at 5,703 |
| `two_stage_shared` | 30,403 per task | 7,968 | 7,968 |

The single-task recipes use two examples per GPU and two accumulation microsteps, giving a global batch of 16. Score Only adapts the original eight-GPU layout to four GPUs while retaining that global batch and update ordering. Shared training uses one scorer and one generator microstep per update: eight examples per task, with an equal average of the two independently token-normalized losses. It starts a fresh optimizer and cosine schedule from scorer checkpoint 5,703. Its save boundaries are fixed update counts.

Checkpoints are saved as `checkpoint-N/`, with Hugging Face model/tokenizer files plus optimizer and per-rank RNG state. Use the table's selected checkpoint for evaluation. Continue an interrupted run with the same recipe, data, output, and initialization arguments plus:

```bash
--resume training/outputs/joint_rubric/checkpoint-1901
```

Resume checks the recipe, data fingerprints, initialization, and world size. `--max-updates N` stops at an absolute update cursor without shortening the learning-rate schedule, and saves a full checkpoint. Checkpoints are written atomically; an interrupted save leaves `.checkpoint-N.incomplete`, which must be removed before retrying that save.

## Small checks

Inspect a recipe without GPUs:

```bash
python -m training.train --recipe two_stage_shared --dry-run
```

Run a short real GPU job before full training:

```bash
python -m training.prepare --output training/smoke-data --limit 32
bash training/launch.sh --recipe joint_rubric --data training/smoke-data \
  --output training/smoke-runs/joint --smoke --max-updates 2
bash training/launch.sh --recipe two_stage_scorer --data training/smoke-data \
  --output training/smoke-runs/scorer --smoke --max-updates 2
bash training/launch.sh --recipe two_stage_shared --data training/smoke-data \
  --output training/smoke-runs/shared --smoke --max-updates 2 \
  --init-from training/smoke-runs/scorer/checkpoint-2
```

Smoke caches and checkpoints are marked separately and rejected by production runs. CPU tests run with `python -m pytest training/tests`; they exercise four-process gradient equivalence and checkpoint continuation. These portable scripts have not yet completed a four-H200 training run or reproduced the paper's numerical results. Historical source references and artifact pins are in [source-provenance.json](source-provenance.json) and [artifacts.json](artifacts.json).
