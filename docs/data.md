# Data construction

The pipeline produces grouped `slates.jsonl` plus candidate-local `per_action.jsonl` for downstream training projections. It does not infer success from an agent's submission: success labels must come from execution-based evaluation.

## Input and collection

For existing Mini-swe-agent outputs, create an index JSONL with one object per trajectory:

```json
{"trajectory_path":"runs/task.traj.json","task_id":"task-id","trajectory_id":"policy-seed-task","task":"Issue text","resolved":true,"replay":{"image":"your-prebuilt-task-image@sha256:...","workdir":"/testbed"},"provenance":{"model":"policy-model-id","revision":"recorded-model-revision","prompt_id":"policy-prompt-hash","source_commit":"source-git-commit"}}
```

Paths are relative to the index. Each native trajectory must contain `messages` and single parsed actions in `message.extra.actions`. Tool-call messages and IDs are preserved; parallel actions are rejected explicitly. Replay images must include the original task checkout and dependencies. Source file hashes are included in resume fingerprints.

```bash
swe-verifier data collect-mini --input trajectories.index.jsonl --output data/01-collected
```

The alternative `collect` command accepts normalized trajectories with the same metadata and `history` plus `steps`, where each step contains `command`, `content`, and `observation: {output, returncode}`. It provides the reference action as the initial candidate, but its observation is deliberately absent until fresh replay. Non-successful trajectories are excluded.

## Construction stages

Prepare a YAML config with `samplers` and `teacher`. Each sampler has an explicit sample count; four additional samples plus the reference action produce a five-candidate group before filtering. Multiple sampler entries support contrasting policy models.

```yaml
samplers:
  - model_name: openai/your-policy-model
    model_class: litellm_textbased
    model_kwargs:
      api_base: http://localhost:8000/v1
      temperature: 1.0
      max_tokens: 2048
    samples: 4
    provenance:
      model: your-policy-model
      revision: recorded-immutable-revision
      prompt_id: original-policy-prompt-hash
      source_commit: source-git-commit
teacher:
  model_name: your-provider/your-teacher-model
  model_kwargs:
    temperature: 0.0
    max_tokens: 8192
  retries: 3
  provenance:
    model: your-teacher-model
    revision: recorded-provider-version
    prompt_id: teacher/annotate.jinja
    source_commit: source-git-commit
```

Set provider credentials in the environment; configs are persisted for provenance and must not contain secrets. Sampling uses Mini’s configured policy model and original history, and accepts one parsed bash action or one bash code block. Set `model_class: litellm` for a native tool-calling policy; use the same policy format and sampling settings as trajectory collection. The teacher uses the packaged paper prompt, receives observed outcomes, and must return one valid prediction/reasoning/score block for every candidate.

```bash
swe-verifier data sample --input data/01-collected/slates.jsonl --output data/02-sampled --config data-config.yaml
swe-verifier data filter --input data/02-sampled/slates.jsonl --output data/03-filtered --threshold 0.8
swe-verifier data replay --input data/03-filtered/slates.jsonl --output data/04-replayed --docker docker --timeout 120
swe-verifier data deduplicate --input data/04-replayed/slates.jsonl --output data/05-deduplicated
swe-verifier data label --input data/05-deduplicated/slates.jsonl --output data/06-labeled --config data-config.yaml
swe-verifier data export --input data/06-labeled/slates.jsonl --output data/final --seed release-v1 --validation-fraction 0.1
swe-verifier data validate --input data/final/slates.jsonl
```

Action filtering retains the first action, then removes actions with regex-token Jaccard similarity **greater than or equal to** the threshold against any retained action. Tokens are lowercased `[A-Za-z0-9_]+`, matching the research training filter. Observation filtering removes duplicate stripped nonempty outputs; distinct actions with empty outputs remain.

Replay creates a fresh Docker container for every candidate, resolves each image to an immutable local image ID included in the resume manifest, replays prior actions, executes only that candidate, records output/exit status, and removes the container. Containers have no host mounts or network access. Images must already be pulled. Prefix replay can expose nondeterminism in tasks; a replayed image and prefix do not guarantee parity with an unavailable historical environment. The Python-only `SnapshotReplay` adapter is for trusted local fixtures and is not a security sandbox.

## Resume and exported contract

Each stage stores a manifest with input, implementation, prompt/config hashes and an atomic cache record per source row. Repeating the same command reuses completed rows. Changed inputs, settings, or code produce an incompatibility error; use a new output directory. A crash during an external call can repeat the unfinished row and its provider charges. Concurrent writers are rejected. The output file is replaced only after all rows complete.

Canonical slates identify the task, trajectory, step, candidate, model/revision, prompt, source commit, replay environment, observations, and teacher annotation. `per_action.jsonl` contains:

- Stable task/trajectory/step/slate/candidate identifiers and a task-level split.
- `messages`: observed policy history followed by one proposed assistant action.
- `target_text`, `score`, shared `rubric`/`summary`, and candidate/teacher provenance.

Teacher observations, future evidence, and raw teacher outputs are excluded from student **input messages**. Predicted observations may appear in the target because that is the supervision objective. Split assignment hashes `(seed, task_id)` before expansion, keeping all trajectories and candidates for a task together. This construction output is separate from the historical checkpoint-training exports. The [training scripts](../training/README.md) consume those pinned exports and verify their native target masks.

Run the synthetic roundtrip without model credentials or Docker:

```bash
python examples/data/offline_roundtrip.py --output /tmp/swe-verifier-data-demo
```

Repeat the command to verify resume. `examples/data/config.yaml` contains the corresponding live-service configuration example. The synthetic fixture records have explicit synthetic provenance and are not paper data.
