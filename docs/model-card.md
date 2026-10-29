# Model artifacts

The release uses Qwen3.5-4B as its base. Trained artifacts already exist on Hugging Face but were **private at inventory time**. The exact immutable revisions and observed visibility are recorded in [the release manifest](../release-manifest.json); a recipe does not make a private checkpoint publicly downloadable.

| Variant | Hugging Face artifact | Roles |
| --- | --- | --- |
| Untrained | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Scalar gate and candidate scorer |
| Score Only | [zkjzou/SWE-Verifier-Score-Only](https://huggingface.co/zkjzou/SWE-Verifier-Score-Only) | Same checkpoint for gate and candidate scorer |
| Joint Rubric | [zkjzou/SWE-Verifier-Joint-Rubric](https://huggingface.co/zkjzou/SWE-Verifier-Joint-Rubric) | Generates rubric and score in one generation |
| Two-Stage Rubric | [zkjzou/SWE-Verifier-Two-Stage-Rubric](https://huggingface.co/zkjzou/SWE-Verifier-Two-Stage-Rubric) | Same MA-7968 weights in separate action-conditioned generator and scorer calls |

Use the pinned revision for the tokenizer and chat template as well as the weights. Native assistant/tool history matters: the proposed action is the final assistant turn, and its execution observation is unavailable to inference. The two-stage scorer receives only the rubric generated for that same candidate.

The scalar score measures suitability of an action; it is not a calibrated probability of task success. Gate-only recipes use the small model for triggering and a separately configured policy model for selection.

## Provenance and license status

The base, Joint Rubric, and Two-Stage Rubric cards declare Apache-2.0. The owner has approved Apache-2.0 for Score Only as well; all three release cards now declare that license. Code licensing does not automatically license weights or dataset contents.

Published checkpoint cards record 30,494 retained examples for Score Only and 30,403 for Joint Rubric; Two-Stage Rubric trains on 30,403 examples in each of its two modes. These are artifact-specific counts, not interchangeable dataset versions. The exact training exports and historical configuration receipts are linked in [the dataset card](dataset-card.md). The [training package](../training/README.md) provides four-H200 launchers, including scorer initialization and shared training for Two-Stage Rubric. Historical optimizer states are not distributed.

## Prepared release revisions

These metadata-only release snapshots retain the original weights and tokenizer bytes. Repositories remain private during preparation.

| Variant | Prepared revision |
| --- | --- |
| score_only | [`73f1413ad911`](https://huggingface.co/zkjzou/SWE-Verifier-Score-Only/tree/73f1413ad9116ed25da0f7af756018b0c818b084) |
| joint_rubric | [`01854c06e9ed`](https://huggingface.co/zkjzou/SWE-Verifier-Joint-Rubric/tree/01854c06e9edf22f0771f9a6d422097e40438055) |
| two_stage | [`9bd79f1eeb76`](https://huggingface.co/zkjzou/SWE-Verifier-Two-Stage-Rubric/tree/9bd79f1eeb762bb9ca1697e5942057067294d87b) |
