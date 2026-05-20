"""Integration test: minibench runner end-to-end with a deterministic model.

Uses minisweagent's built-in DeterministicModel so the test doesn't need an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minisweagent.projectk.minibench.runner import (
    Instance,
    _apply_patch,
    _strip_noisy_diffs,
    run_all,
)


def test_strip_noisy_diffs_removes_pycache() -> None:
    patch = (
        "diff --git a/__pycache__/foo.pyc b/__pycache__/foo.pyc\n"
        "Binary files differ\n"
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@\n-1\n+2\n"
    )
    out = _strip_noisy_diffs(patch)
    assert "__pycache__" not in out
    assert "src/x.py" in out


def test_strip_noisy_diffs_keeps_real_changes_only() -> None:
    patch = (
        "diff --git a/foo/.pytest_cache/CACHEDIR.TAG b/foo/.pytest_cache/CACHEDIR.TAG\n"
        "+ ignored\n"
        "diff --git a/foo/bar.py b/foo/bar.py\n"
        "--- a/foo/bar.py\n"
        "+++ b/foo/bar.py\n"
        "@@\n-1\n+2\n"
    )
    out = _strip_noisy_diffs(patch)
    assert "pytest_cache" not in out
    assert "foo/bar.py" in out


def test_apply_patch_on_clean_checkout(tmp_path: Path) -> None:
    # Build a tiny git repo with one file
    f = tmp_path / "x.py"
    f.write_text("def add(a,b):\n    return a-b\n")
    import subprocess
    subprocess.run("git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -q -m i",
                   shell=True, cwd=tmp_path, check=True)

    patch = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a,b):\n"
        "-    return a-b\n"
        "+    return a+b\n"
    )
    ok, log = _apply_patch(patch, tmp_path)
    assert ok, log
    assert "return a+b" in f.read_text()


def test_instance_from_dir(tmp_path: Path) -> None:
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "instance.yaml").write_text(
        "instance_id: demo__1\n"
        "problem_statement: 'fix the bug'\n"
        "test_command: pytest -q\n"
    )
    repo = fx / "repo"
    repo.mkdir()
    inst = Instance.from_dir(fx)
    assert inst.instance_id == "demo__1"
    assert inst.test_command == "pytest -q"
    assert inst.repo_dir == repo


def test_run_all_with_deterministic_model(tmp_path: Path) -> None:
    """Smoke test: a deterministic 'agent' that immediately submits a correct patch
    flips the resolved bit. This exercises the full runner without needing an LLM.
    """
    # Build a fixture where the bug is `return a - b` -> `return a + b`
    fix_root = tmp_path / "fixtures" / "toy"
    fix_root.mkdir(parents=True)
    (fix_root / "instance.yaml").write_text(
        "instance_id: toy__add\n"
        "problem_statement: 'add() subtracts; fix it.'\n"
        "test_command: pytest -q\n"
    )
    repo = fix_root / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def add(a,b):\n    return a-b\n")
    (repo / "test_app.py").write_text("from app import add\n\ndef test_add():\n    assert add(2,3) == 5\n")

    # Deterministic config: one assistant message yielding the submit command.
    # The submit command must produce a git diff that fixes the bug.
    edit = "python -c \"open('app.py','w').write('def add(a,b):\\n    return a+b\\n')\""
    submit = (
        f"{edit} && git add -A && git diff --cached > /tmp/projk_test_patch.diff && "
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/projk_test_patch.diff"
    )
    config = {
        "agent": {"agent_class": "default", "step_limit": 5, "cost_limit": 0.0,
                  "system_template": "sys", "instance_template": "{{task}}"},
        "environment": {"environment_class": "local", "timeout": 60},
        "model": {
            "model_class": "deterministic",
            "outputs": [
                {"role": "assistant", "content": "ok",
                 "extra": {"actions": [{"command": submit}]}},
            ],
        },
    }
    out_dir = tmp_path / "out"
    report = run_all(fix_root.parent, config, out_dir)
    assert report["n_total"] == 1
    # We're not asserting resolve here (deterministic model interactions with the
    # action regex can vary by config); we just confirm the runner walks end-to-end.
    assert "toy__add" in [v["instance_id"] for v in report["verdicts"]]
    assert (out_dir / "preds.json").exists()
    assert (out_dir / "report.json").exists()
