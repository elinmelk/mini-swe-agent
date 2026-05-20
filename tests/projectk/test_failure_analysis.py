"""Unit tests for minisweagent.projectk.failure_analysis."""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.projectk.failure_analysis import (
    analyze_directory,
    classify_trajectory,
)


def _traj(messages, exit_status="Submitted", submission="diff --git a/x.py b/x.py\n", model_stats=None):
    return {
        "info": {
            "model_stats": model_stats or {"api_calls": len(messages), "instance_cost": 0.0},
            "exit_status": exit_status,
            "submission": submission,
        },
        "messages": messages,
    }


def _write(path: Path, traj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(traj))
    return path


def test_resolved_bucket(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.traj.json", _traj([]))
    r = classify_trajectory(p, submission="diff --git a/x.py b/x.py\n", resolved=True)
    assert r.label == "RESOLVED"


def test_budget_exceeded_bucket(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.traj.json", _traj([], exit_status="LimitsExceeded", submission=""))
    r = classify_trajectory(p, submission="", resolved=False)
    assert r.label == "BUDGET_EXCEEDED"


def test_infinite_loop_bucket(tmp_path: Path) -> None:
    cmd = {"command": "ls"}
    msgs = []
    for _ in range(6):
        msgs.append({"role": "assistant", "content": "", "extra": {"actions": [cmd]}})
        msgs.append({"role": "user", "content": "<returncode>0</returncode>"})
    p = _write(tmp_path / "x.traj.json", _traj(msgs, exit_status="LimitsExceeded", submission="diff --git a/y.py b/y.py\n"))
    r = classify_trajectory(p, submission="diff --git a/y.py b/y.py\n", resolved=False)
    # LimitsExceeded short-circuits BEFORE the loop detector — that's by design.
    assert r.label == "BUDGET_EXCEEDED"

    # Now without LimitsExceeded: loop detector should fire
    p2 = _write(tmp_path / "y.traj.json", _traj(msgs, exit_status="Submitted", submission="diff --git a/y.py b/y.py\n"))
    r2 = classify_trajectory(p2, submission="diff --git a/y.py b/y.py\n", resolved=False)
    assert r2.label == "INFINITE_LOOP"


def test_no_patch_bucket(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.traj.json", _traj([], submission=""))
    r = classify_trajectory(p, submission="", resolved=False)
    assert r.label == "NO_PATCH"


def test_wrong_file_edited_bucket(tmp_path: Path) -> None:
    patch = "diff --git a/tests/test_foo.py b/tests/test_foo.py\n@@\n-1\n+2\n"
    p = _write(tmp_path / "x.traj.json", _traj([], submission=patch))
    r = classify_trajectory(p, submission=patch, resolved=False)
    assert r.label == "WRONG_FILE_EDITED"


def test_tests_broken_bucket(tmp_path: Path) -> None:
    msgs = [
        {"role": "assistant", "content": "", "extra": {"actions": [{"command": "pytest -q"}]}},
        {"role": "user", "content": "<returncode>1</returncode>"},
        {"role": "assistant", "content": "", "extra": {"actions": [{"command": "pytest -q"}]}},
        {"role": "user", "content": "<returncode>1</returncode>"},
        {"role": "assistant", "content": "", "extra": {"actions": [{"command": "pytest -q"}]}},
        {"role": "user", "content": "<returncode>1</returncode>"},
    ]
    patch = "diff --git a/x.py b/x.py\n@@\n-1\n+2\n"
    p = _write(tmp_path / "x.traj.json", _traj(msgs, submission=patch))
    r = classify_trajectory(p, submission=patch, resolved=False)
    assert r.label == "TESTS_BROKEN"


def test_probably_wrong_edit_bucket(tmp_path: Path) -> None:
    msgs = [
        {"role": "assistant", "content": "", "extra": {"actions": [{"command": "pytest -q"}]}},
        {"role": "user", "content": "<returncode>0</returncode>"},
    ]
    patch = "diff --git a/x.py b/x.py\n@@\n-1\n+2\n"
    p = _write(tmp_path / "x.traj.json", _traj(msgs, submission=patch))
    r = classify_trajectory(p, submission=patch, resolved=False)
    assert r.label == "PROBABLY_WRONG_EDIT"


def test_analyze_directory_summary(tmp_path: Path) -> None:
    # Two instances: one resolved, one with no patch
    a = tmp_path / "a"
    a.mkdir()
    _write(a / "a.traj.json", _traj([], submission="diff --git a/x.py b/x.py\n"))
    b = tmp_path / "b"
    b.mkdir()
    _write(b / "b.traj.json", _traj([], submission="", exit_status="Submitted"))
    (tmp_path / "preds.json").write_text(json.dumps({
        "a": {"instance_id": "a", "model_patch": "diff --git a/x.py b/x.py\n", "model_name_or_path": "m"},
        "b": {"instance_id": "b", "model_patch": "", "model_name_or_path": "m"},
    }))
    out = analyze_directory(tmp_path, resolved_ids={"a"})
    assert out["summary"]["RESOLVED"] == 1
    assert out["summary"]["NO_PATCH"] == 1
    assert len(out["records"]) == 2
