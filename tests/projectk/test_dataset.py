"""Unit tests for minisweagent.projectk.dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from minisweagent.projectk.dataset import curated_lite_slice, load_jsonl_instances


def _mk(instance_id: str, repo: str) -> dict:
    return {"instance_id": instance_id, "repo": repo, "problem_statement": "x"}


def test_curated_slice_is_deterministic() -> None:
    instances = [_mk(f"r{r}__i{i}", f"r{r}") for r in range(4) for i in range(10)]
    a = [x["instance_id"] for x in curated_lite_slice(instances, n=12, seed=42)]
    b = [x["instance_id"] for x in curated_lite_slice(instances, n=12, seed=42)]
    assert a == b


def test_curated_slice_balances_across_repos() -> None:
    instances = [_mk(f"r{r}__i{i}", f"r{r}") for r in range(4) for i in range(10)]
    chosen = curated_lite_slice(instances, n=8, seed=42)
    assert len(chosen) == 8
    by_repo = Counter(x["repo"] for x in chosen)
    # With 8 picks across 4 repos round-robin, no repo should hog more than 3
    assert max(by_repo.values()) - min(by_repo.values()) <= 1


def test_curated_slice_does_not_exceed_n() -> None:
    instances = [_mk(f"r0__i{i}", "r0") for i in range(3)]
    chosen = curated_lite_slice(instances, n=10, seed=42)
    # Only 3 instances available — should return all 3, not loop forever
    assert len(chosen) == 3


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "data.jsonl"
    rows = [{"instance_id": "a", "repo": "x"}, {"instance_id": "b", "repo": "y"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = load_jsonl_instances(p)
    assert out == rows
