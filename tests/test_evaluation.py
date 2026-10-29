"""Evaluation must retain failed attempts and distinguish incomplete harness reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_verifier.evaluation import summarize_runs


def write_receipts(root: Path, statuses: dict[str, str]) -> None:
    for instance_id, status in statuses.items():
        directory = root / instance_id
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "recipe_name": "recipe",
                    "status": status,
                    "selected_steps": 1,
                    "generated_steps": 1,
                    "verified_steps": 0,
                    "usage": [{"role": "policy", "cost_usd": 0.25, "latency_seconds": 1.0}],
                }
            )
        )


@pytest.mark.parametrize("unsubmitted", [[], ["outside-selected-subset"]])
def test_failed_attempt_and_cost_remain_in_denominator(tmp_path: Path, unsubmitted: list[str]) -> None:
    runs = tmp_path / "runs"
    write_receipts(runs, {"solved": "completed", "crashed": "failed"})
    ungraded = summarize_runs(runs, None)
    assert ungraded["grading_status"] == "not_graded"
    assert "resolve_rate" not in ungraded
    assert ungraded["grading_setup"]["repository"] == "https://github.com/SWE-bench/SWE-bench"
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "submitted_ids": ["solved", "crashed"],
                "resolved_ids": ["solved"],
                "unresolved_ids": [],
                "completed_ids": ["solved"],
                "empty_patch_ids": ["crashed"],
                "error_ids": [],
                "incomplete_ids": unsubmitted,
            }
        )
    )
    summary = summarize_runs(runs, report)
    assert summary["grading_status"] == "external_report_imported"
    assert summary["instances"] == 2
    assert summary["failed_runs"] == 1
    assert summary["known_cost_usd"] == 0.5
    assert summary["resolve_rate"] == 0.5
    assert summary["evaluation_complete"] is True


def test_partial_report_is_explicit_lower_bound_not_complete_rate(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_receipts(runs, {name: "completed" for name in ("solved", "error", "pending", "unknown")})
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "submitted_ids": ["solved", "error", "pending", "unknown"],
                "resolved_ids": ["solved"],
                "unresolved_ids": [],
                "completed_ids": ["solved"],
                "error_ids": ["error"],
                "incomplete_ids": ["pending"],
                "empty_patch_ids": [],
            }
        )
    )
    summary = summarize_runs(runs, report)
    assert summary["resolve_rate"] is None
    assert summary["attempted_resolve_rate_lower_bound"] == 0.25
    assert summary["evaluation_error_instances"] == 1
    assert summary["evaluation_incomplete_instances"] == 1
    assert summary["evaluation_unknown_instances"] == 1
    assert summary["evaluation_complete"] is False


def test_wrong_group_or_ambiguous_report_is_rejected(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_receipts(runs, {"own-task": "completed"})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"resolved_ids": ["own-task"]}))
    with pytest.raises(ValueError, match="submitted_ids"):
        summarize_runs(runs, report)
    report.write_text(
        json.dumps({"submitted_ids": ["other-task"], "resolved_ids": ["other-task"], "unresolved_ids": []})
    )
    with pytest.raises(ValueError, match="does not cover"):
        summarize_runs(runs, report)
    report.write_text(
        json.dumps({"submitted_ids": ["own-task"], "resolved_ids": ["own-task"], "unresolved_ids": ["own-task"]})
    )
    with pytest.raises(ValueError, match="overlapping"):
        summarize_runs(runs, report)


def test_resolved_report_cannot_match_failed_empty_prediction(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_receipts(runs, {"task": "failed"})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"submitted_ids": ["task"], "resolved_ids": ["task"], "unresolved_ids": []}))
    with pytest.raises(ValueError, match="failed run receipt"):
        summarize_runs(runs, report)
