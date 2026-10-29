# SWE-Verifier training data

Training data for the three trained models used in [SWE-Verifier](https://arxiv.org/abs/2610.30284v1).

## Models

| Paper variant | Hugging Face model |
| --- | --- |
| Score Only | [zkjzou/SWE-Verifier-Score-Only](https://huggingface.co/zkjzou/SWE-Verifier-Score-Only) |
| Joint Rubric | [zkjzou/SWE-Verifier-Joint-Rubric](https://huggingface.co/zkjzou/SWE-Verifier-Joint-Rubric) |
| Two-Stage Rubric | [zkjzou/SWE-Verifier-Two-Stage-Rubric](https://huggingface.co/zkjzou/SWE-Verifier-Two-Stage-Rubric) |

## Data

| Configuration | Rows | Split |
| --- | ---: | --- |
| `score_only` | 30,494 | `train` |
| `joint_rubric` | 30,403 | `train` |
| `two_stage_generator` | 30,403 | `train` |
| `two_stage_scorer` | 30,403 | `train` |
| `raw_teacher` | 6,669 request/response pairs | `corpus` |

Training rows contain `messages`, with the final assistant message as the target. The raw teacher pairs cover 33,343 candidate actions. They contain saved teacher inputs and output text; provider metadata is excluded, and detected sensitive patterns are redacted. The `redacted` field identifies affected rows.

```python
from datasets import load_dataset

training = load_dataset("zkjzou/swe-verifier-training", "score_only", split="train")
raw = load_dataset("zkjzou/swe-verifier-training", "raw_teacher", split="corpus", streaming=True)
```

Decode `input_json` with `json.loads` to read a teacher input. `teacher_raw_output` contains its output text. Action identifiers link raw records to the training manifests.

The repository is private during release preparation; an authorized Hugging Face login is required.

## License

Original annotations and packaging: Apache 2.0. Embedded upstream material retains its original licenses; notices are included in `upstream_licenses/`.

[Paper](https://arxiv.org/abs/2610.30284v1) · [Code](https://github.com/zkjzou/SWE-Verifier)
