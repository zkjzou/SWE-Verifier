"""Import external SWE-bench reports; no grading server or test harness runs here."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def _ids(outcomes: Mapping[str, object], field: str) -> set[str]:
    value = outcomes.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"Official evaluation report requires a list of instance IDs in {field}")
    if len(value) != len(set(value)):
        raise ValueError(f"Official evaluation report contains duplicate {field}")
    return set(value)


def _outcomes(report: Path, instance_ids: set[str], failed_ids: set[str]) -> dict[str, object]:
    raw = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected an official SWE-bench aggregate report object")
    submitted = _ids(raw, "submitted_ids")
    resolved = _ids(raw, "resolved_ids")
    unresolved = _ids(raw, "unresolved_ids")
    categories = {
        "resolved": resolved,
        "unresolved": unresolved,
        "empty_patch": _ids(raw, "empty_patch_ids") if "empty_patch_ids" in raw else set(),
        "error": _ids(raw, "error_ids") if "error_ids" in raw else set(),
        # SWE-bench reports list unsubmitted dataset tasks as incomplete too.
        # Only submitted tasks can be pending outcomes for this evaluation.
        "incomplete": (_ids(raw, "incomplete_ids") & submitted) if "incomplete_ids" in raw else set(),
    }
    observed: set[str] = set()
    for category, values in categories.items():
        if observed & values:
            raise ValueError(f"Official report assigns overlapping outcome categories: {category}")
        if values - submitted:
            raise ValueError(f"Official report has {category} outcomes outside submitted_ids")
        observed |= values
    if "completed_ids" in raw and _ids(raw, "completed_ids") != resolved | unresolved:
        raise ValueError("Official report completed_ids must match resolved_ids plus unresolved_ids")
    missing = instance_ids - submitted
    if missing:
        raise ValueError(f"Evaluation report does not cover run receipts: {', '.join(sorted(missing))}")
    if failed_ids & resolved:
        raise ValueError(
            "A failed run receipt cannot have a resolved patch; verify that the report matches these predictions"
        )
    counts = {category: len(values & instance_ids) for category, values in categories.items()}
    unknown = len(instance_ids - observed)
    complete = counts["error"] == counts["incomplete"] == unknown == 0
    lower_bound = counts["resolved"] / len(instance_ids)
    return {
        "resolved": counts["resolved"],
        "unresolved": counts["unresolved"],
        "empty_patch_instances": counts["empty_patch"],
        "evaluation_error_instances": counts["error"],
        "evaluation_incomplete_instances": counts["incomplete"],
        "evaluation_unknown_instances": unknown,
        "evaluation_complete": complete,
        "resolve_rate": lower_bound if complete else None,
        "attempted_resolve_rate_lower_bound": lower_bound,
        "evaluation_report": str(report),
    }


def summarize_runs(root: Path, report: Path | None) -> dict[str, object]:
    paths = sorted(root.rglob("run.json"))
    if not paths:
        raise ValueError(f"No run.json receipts under {root}")
    rows: list[dict[str, object]] = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each run receipt must be an object")
    ids = [str(row["instance_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate instance IDs: summarize each recipe and seed in its own run directory")
    if len({str(row["recipe_name"]) for row in rows}) != 1:
        raise ValueError("A summary must contain one recipe; separate experiment output directories")
    statuses = [row.get("status") for row in rows]
    if any(status not in {"completed", "failed"} for status in statuses):
        raise ValueError("Every run receipt must explicitly record status completed or failed")
    failed_ids = {str(row["instance_id"]) for row in rows if row["status"] == "failed"}
    for row in rows:
        for field in ("selected_steps", "generated_steps", "verified_steps"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"Run receipt {field} must be a nonnegative integer")
        if row["verified_steps"] > row["selected_steps"]:
            raise ValueError("Run receipt verified_steps cannot exceed selected_steps")
        if not isinstance(row.get("usage"), list):
            raise ValueError("Run receipt usage must be a list")
    usage = [cast(dict[str, object], u) for row in rows for u in cast(list[object], row["usage"])]
    if any(not isinstance(item, dict) for item in usage):
        raise ValueError("Usage entries must be objects")
    roles = Counter(str(u["role"]) for u in usage)
    costs: list[float] = []
    for item in usage:
        cost = item.get("cost_usd")
        if cost is not None:
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                raise ValueError("Receipt cost_usd must be finite and nonnegative or null")
            costs.append(float(cost))
    result: dict[str, object] = {
        "instances": len(rows),
        "recipe": rows[0]["recipe_name"],
        "completed_runs": len(rows) - len(failed_ids),
        "failed_runs": len(failed_ids),
        "selected_steps_mean": sum(int(row["selected_steps"]) for row in rows) / len(rows),
        "generated_steps": sum(int(row["generated_steps"]) for row in rows),
        "verified_steps": sum(int(row["verified_steps"]) for row in rows),
        "calls_by_role": dict(roles),
        "known_cost_usd": sum(costs),
        "cost_complete": len(costs) == len(usage),
        "model_latency_seconds": sum(float(u["latency_seconds"]) for u in usage),
        "pricing_note": "Unknown local serving prices remain null; known_cost_usd is not total compute cost.",
    }
    steps = sum(int(row["selected_steps"]) for row in rows)
    result["verify_rate"] = result["verified_steps"] / steps if steps else None
    if report is not None:
        result.update(_outcomes(report, set(ids), failed_ids))
        result["grading_status"] = "external_report_imported"
    else:
        result["grading_status"] = "not_graded"
        result["grading_setup"] = {
            "repository": "https://github.com/SWE-bench/SWE-bench",
            "instructions": "https://www.swebench.com/SWE-bench/guides/evaluation/",
            "next_step": "Grade exported predictions.jsonl externally, then pass the aggregate JSON report with --report.",
        }
    return result
