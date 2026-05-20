"""Unit tests for minisweagent.projectk.metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minisweagent.projectk.metrics import compute_metrics, load_resolved_from_report


def _write_traj(path: Path, *, api_calls: int, cost: float, exit_status: str,
                submission: str, usages: list[tuple[int, int]],
                timestamps: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = []
    for i, (pt, ct) in enumerate(usages):
        messages.append({
            "role": "assistant",
            "content": "",
            "extra": {
                "actions": [{"command": "ls"}],
                "timestamp": timestamps[i],
                "response": {"usage": {"prompt_tokens": pt, "completion_tokens": ct}},
            },
        })
        messages.append({"role": "user", "content": "<returncode>0</returncode>"})
    path.write_text(json.dumps({
        "info": {
            "model_stats": {"api_calls": api_calls, "instance_cost": cost},
            "exit_status": exit_status,
            "submission": submission,
        },
        "messages": messages,
    }))


def test_compute_metrics_aggregates_correctly(tmp_path: Path) -> None:
    # Two instances: one submitted+resolved, one budget-exceeded with empty patch
    inst1 = tmp_path / "demo__a-1"
    _write_traj(
        inst1 / "demo__a-1.traj.json",
        api_calls=3, cost=0.012, exit_status="Submitted",
        submission="diff --git a/x.py b/x.py\n",
        usages=[(100, 10), (50, 5), (40, 4)],
        timestamps=[1.0, 1.5, 2.0],
    )
    inst2 = tmp_path / "demo__b-2"
    _write_traj(
        inst2 / "demo__b-2.traj.json",
        api_calls=30, cost=0.05, exit_status="LimitsExceeded",
        submission="",
        usages=[(80, 8)] * 4,
        timestamps=[10.0, 11.0, 12.0, 13.0],
    )
    (tmp_path / "preds.json").write_text(json.dumps({
        "demo__a-1": {"instance_id": "demo__a-1", "model_patch": "diff --git ...\n", "model_name_or_path": "m"},
        "demo__b-2": {"instance_id": "demo__b-2", "model_patch": "", "model_name_or_path": "m"},
    }))

    m = compute_metrics(tmp_path, resolved_instances={"demo__a-1"})

    assert m.n_attempts == 2
    assert m.n_submitted == 1
    assert m.n_resolved == 1
    assert m.resolve_rate == pytest.approx(0.5)
    assert m.submission_rate == pytest.approx(0.5)
    assert m.mean_tool_calls == pytest.approx((3 + 30) / 2)
    assert m.mean_cost_usd == pytest.approx((0.012 + 0.05) / 2)
    assert m.mean_input_tokens == pytest.approx((190 + 320) / 2)
    assert m.mean_output_tokens == pytest.approx((19 + 32) / 2)
    assert m.exit_status_counts == {"Submitted": 1, "LimitsExceeded": 1}
    assert len(m.per_instance) == 2
    assert m.per_instance[0]["instance_id"] in {"demo__a-1", "demo__b-2"}


def test_compute_metrics_handles_empty_dir(tmp_path: Path) -> None:
    m = compute_metrics(tmp_path)
    assert m.n_attempts == 0
    assert m.resolve_rate == 0.0
    assert m.exit_status_counts == {}


def test_load_resolved_from_report_resolved_ids_list(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"resolved_ids": ["a", "b", "c"]}))
    assert load_resolved_from_report(p) == {"a", "b", "c"}


def test_load_resolved_from_report_per_instance_resolved_flag(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "x": {"resolved": True},
        "y": {"resolved": False},
        "z": {"resolved": True},
    }))
    assert load_resolved_from_report(p) == {"x", "z"}
