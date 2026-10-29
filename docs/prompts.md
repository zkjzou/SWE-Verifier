# Prompt inventory

Prompts are installed package resources. `load_prompt(name)` verifies the registered checksum; `render(name, variables)` uses Jinja StrictUndefined so missing fields fail explicitly. Source references, original source hashes, extraction rules, and packaged hashes are in `swe_verifier/prompts/index.json`.

## Families

| Family | Purpose and contract |
| --- | --- |
| `policy` | Original policy system/task, observation, and format-error templates recovered from a completed full500 run. |
| `untrained` | Single-action scalar gate/scorer. Uses task, candidate, native history; emits REASONING and SCORE. |
| `score_only` | Same scalar interface as Untrained; trained checkpoint changes the model, not the scoring template. |
| `joint_rubric` | Exact checkpoint system/scoring/retry text; native task/history/candidate messages; emits rubric and terminal SCORE. |
| `two_stage` | Exact checkpoint generator and scorer modes; scorer.jinja requires checklist_text. Each proposed action has its own generated rubric. |
| `selection` | Policy-model multi-action comparison for all-step and gate-only; candidates have index/content and FINAL is one-based. |
| `teacher` | Exact final sol_64k_v1 teacher system/template; candidate observations are available only to teacher annotation. |
| `trajectory` | Exact criterion pairwise prompt and criteria constants from the trajectory baseline; score_A/score_B use A–T. |
| `analysis` | Action-category/phase classifier prompts used in paper analyses, separate from the score-threshold gate. |
| `paper` | Verbatim prompt blocks from the manuscript for comparison; these are archival resources, not implicit runtime defaults. |

The packaged index is the canonical list of prompt inputs, output contracts, recipes, attribution, and checksums.
