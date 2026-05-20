"""Classify per-instance trajectories into failure buckets.

The Project K writeup asks for an explicit failure taxonomy. We look at each
unresolved trajectory and assign exactly one label from:

  RESOLVED                # baseline: don't touch this one
  BUDGET_EXCEEDED         # LimitsExceeded
  INFINITE_LOOP           # >= 5 identical consecutive commands or format errors
  NO_PATCH                # agent exited but produced an empty patch
  TESTS_BROKEN            # tests run by the agent ended with a non-zero return
  WRONG_FILE_EDITED       # patch touches only test/build files, not source
  PROBABLY_WRONG_EDIT     # submitted a patch, tests didn't pass / not resolved
  CRASH                   # uncaught exception during the run
  OTHER

We optionally accept a `resolved_ids` set so we can split "submitted but wrong
edit" from "actually resolved".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TEST_PATH_RE = re.compile(r"(^|/)(tests?|conftest\.py|setup\.(py|cfg)|pyproject\.toml|tox\.ini)")
_PATCH_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)


@dataclass
class FailureRecord:
    instance_id: str
    label: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"instance_id": self.instance_id, "label": self.label, "detail": self.detail}


def _commands(traj: dict) -> list[str]:
    cmds: list[str] = []
    for msg in traj.get("messages", []):
        for action in (msg.get("extra") or {}).get("actions") or []:
            cmd = action.get("command")
            if isinstance(cmd, str):
                cmds.append(cmd.strip())
    return cmds


def _has_command_loop(commands: list[str], window: int = 5) -> bool:
    if len(commands) < window:
        return False
    for i in range(len(commands) - window + 1):
        if len(set(commands[i : i + window])) == 1:
            return True
    return False


def _test_returncodes(traj: dict) -> list[int]:
    codes: list[int] = []
    for i, msg in enumerate(traj.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        for action in (msg.get("extra") or {}).get("actions") or []:
            cmd = (action.get("command") or "").lower()
            if "pytest" in cmd or "python -m unittest" in cmd or "tox" in cmd:
                if i + 1 < len(traj["messages"]):
                    obs = traj["messages"][i + 1].get("content") or ""
                    m = re.search(r"<returncode>(-?\d+)</returncode>", obs)
                    if m:
                        codes.append(int(m.group(1)))
    return codes


def _patch_only_touches_tests(patch: str) -> bool:
    files = _PATCH_FILE_RE.findall(patch)
    if not files:
        return False
    return all(_TEST_PATH_RE.search(f) for f in files)


def classify_trajectory(traj_path: Path, *, submission: str, resolved: bool) -> FailureRecord:
    instance_id = traj_path.stem.removesuffix(".traj")
    try:
        traj = json.loads(traj_path.read_text())
    except Exception as e:
        return FailureRecord(instance_id, "CRASH", f"could not read trajectory: {e}")

    info = traj.get("info", {})
    exit_status = info.get("exit_status") or ""

    if resolved:
        return FailureRecord(instance_id, "RESOLVED", "")

    if exit_status == "LimitsExceeded":
        return FailureRecord(instance_id, "BUDGET_EXCEEDED", "step or cost cap hit")

    if exit_status and exit_status not in {"Submitted", "Exit", ""}:
        # Anything that's not a normal exit and isn't a budget cap is treated as a crash.
        return FailureRecord(instance_id, "CRASH", f"exit_status={exit_status}")

    commands = _commands(traj)
    if _has_command_loop(commands):
        return FailureRecord(instance_id, "INFINITE_LOOP", "5+ identical consecutive commands")

    if not submission.strip():
        return FailureRecord(instance_id, "NO_PATCH", "agent finished without producing a patch")

    if _patch_only_touches_tests(submission):
        return FailureRecord(instance_id, "WRONG_FILE_EDITED", "patch only touches tests/build files")

    test_codes = _test_returncodes(traj)
    if test_codes and all(c != 0 for c in test_codes[-3:]):
        return FailureRecord(instance_id, "TESTS_BROKEN", f"last test returncodes: {test_codes[-3:]}")

    return FailureRecord(instance_id, "PROBABLY_WRONG_EDIT", "patch submitted but not resolved")


def analyze_directory(output_dir: str | Path, resolved_ids: set[str] | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    resolved_ids = resolved_ids or set()
    preds_path = output_dir / "preds.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}

    records: list[FailureRecord] = []
    for inst_dir in sorted(output_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        traj_path = inst_dir / f"{inst_dir.name}.traj.json"
        if not traj_path.exists():
            continue
        submission = (preds.get(inst_dir.name) or {}).get("model_patch") or ""
        records.append(
            classify_trajectory(traj_path, submission=submission, resolved=inst_dir.name in resolved_ids)
        )

    counts: Counter[str] = Counter(r.label for r in records)
    return {
        "summary": dict(counts),
        "records": [r.to_dict() for r in records],
    }
