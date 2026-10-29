from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_container_start_failure_keeps_attempt_and_input_provenance(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text(json.dumps({"instance_id": "fixture__task-1", "problem_statement": "Read the fixture."}))
    output = tmp_path / "attempt"
    environment = dict(os.environ)
    environment["MSWEA_SILENT_STARTUP"] = "1"
    environment["MSWEA_GLOBAL_CONFIG_DIR"] = str(tmp_path / "mini-config")
    process = subprocess.run(
        [sys.executable, "-m", "swe_verifier", "run", "--recipe", "none", "--instance-json", str(instance),
         "--docker-executable", str(tmp_path / "missing-container-runtime"), "--output", str(output)],
        env=environment, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=60,
    )
    assert process.returncode != 0
    receipt = json.loads((output / "run.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "FileNotFoundError"
    assert receipt["usage"] == []
    assert receipt["dataset"] is None
    assert receipt["dataset_revision"] is None
    assert receipt["input_source"] == "instance_json"
    assert receipt["input_sha256"] == hashlib.sha256(instance.read_bytes()).hexdigest()
    assert json.loads((output / "predictions.jsonl").read_text())["model_patch"] == ""
