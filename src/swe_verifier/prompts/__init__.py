"""Load registered release prompts from the installed package, independent of cwd."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import TypedDict, cast

from jinja2 import Environment, StrictUndefined


class PromptMessageContext(TypedDict, total=False):
    message_context: list[str]


class PromptRecord(PromptMessageContext):
    name: str
    sha256: str
    source: str
    source_sha256: str
    extraction: str
    purpose: str
    output_contract: str
    recipes: list[str]
    input_variables: list[str]
    paper_status: str


def prompt_index() -> list[PromptRecord]:
    """Return the source and checksum inventory for every packaged template."""
    data = json.loads(files(__package__).joinpath("index.json").read_text(encoding="utf-8"))
    return cast(list[PromptRecord], data["prompts"])


def load_prompt(name: str) -> str:
    """Read a registered template and reject checksum mismatches or unknown paths."""
    records = {record["name"]: record for record in prompt_index()}
    if name not in records:
        raise ValueError(f"Unknown SWE-Verifier prompt: {name!r}")
    content = files(__package__).joinpath(*name.split("/")).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != records[name]["sha256"]:
        raise ValueError(f"Packaged prompt checksum mismatch: {name!r}")
    return content.decode("utf-8")


def render(name: str, variables: Mapping[str, object]) -> str:
    """Render an exact template, rejecting missing variables instead of guessing."""
    environment = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    return environment.from_string(load_prompt(name)).render(dict(variables))
