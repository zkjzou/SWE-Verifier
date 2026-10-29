"""Validated, installable recipes with independent policy, gate, and selector roles."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from swe_verifier.inference.types import Variant


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    revision: str | None
    backend: Literal["native", "litellm"]
    kwargs: dict[str, object]
    retries: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0)


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    description: str
    strategy: Literal["none", "all_step", "selective", "trajectory"]
    variant: Variant
    selector_role: Literal["none", "verifier", "policy"]
    threshold: float = Field(ge=0, le=1)
    candidate_count: int = Field(ge=1)
    scoring_draws: int = Field(default=3, ge=1)
    format_retries: int = Field(ge=0)
    failure_policy: Literal["raise", "first_valid"]
    step_limit: int = Field(gt=0)
    max_calls: int = Field(gt=0)
    cost_limit_usd: float | None = Field(gt=0)
    policy_cost_limit_usd: float = Field(ge=0)
    wall_time_limit_seconds: int = Field(ge=0)
    rollout_count: int = Field(gt=0)
    trajectory_repetitions: int = Field(gt=0)
    policy: ModelSettings
    verifier: ModelSettings | None
    provenance_status: str

    @model_validator(mode="after")
    def validate_roles(self) -> Recipe:
        if self.policy.backend != "litellm":
            raise ValueError("Policy actions currently require the Mini LiteLLM backend; native is verifier-only")
        if self.strategy == "selective" and self.verifier is None:
            raise ValueError("Selective verification requires an explicit verifier model")
        if self.selector_role == "verifier" and self.verifier is None:
            raise ValueError("Verifier selection requires an explicit verifier model")
        if self.strategy in {"selective", "all_step"} and self.selector_role == "none":
            raise ValueError("Action comparison requires a selector")
        if self.verifier is not None and self.verifier.backend == "native":
            revision = self.verifier.revision
            if revision is None or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("Native checkpoints require an immutable 40-character revision")
        return self


def recipe_names() -> tuple[str, ...]:
    return tuple(sorted(path.name.removesuffix(".yaml") for path in files("swe_verifier.recipes").iterdir()
                        if path.name.endswith(".yaml")))


def load_recipe(identifier: str) -> Recipe:
    candidate = Path(identifier)
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    else:
        if identifier not in recipe_names():
            raise ValueError(f"Unknown recipe {identifier!r}; available: {', '.join(recipe_names())}")
        text = files("swe_verifier.recipes").joinpath(f"{identifier}.yaml").read_text(encoding="utf-8")
    return Recipe.model_validate(yaml.safe_load(text))
