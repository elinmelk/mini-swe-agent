"""Best-of-N rollouts with test-based verifier.

For each minibench fixture, run the agent N times (independent trajectories,
different RNG / sampling seeds), apply each candidate patch to a fresh
checkout, run the test command, and pick the trajectory whose patch leaves
the fewest test failures (ties broken by smallest patch).

This is the technique that powers Agentless's resolve-rate boost and SWE-Gym's
verifier training signal. The verifier is the test command itself — no
learned scorer, no separate model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from minisweagent.projectk.minibench.runner import (
    Instance,
    _apply_patch,
    _materialize,
    _run as _shell_run,
    discover_fixtures,
    run_instance,
)


@dataclass
class CandidateScore:
    trajectory_dir: Path
    patch: str
    tests_pass: bool
    failing_tests: int
    patch_size: int

    def as_dict(self) -> dict:
        return {
            "trajectory_dir": str(self.trajectory_dir),
            "tests_pass": self.tests_pass,
            "failing_tests": self.failing_tests,
            "patch_size": self.patch_size,
        }


def _count_failing(test_output: str) -> int:
    """Best-effort failing-test count from pytest output."""
    import re
    m = re.search(r"(\d+)\s+failed", test_output)
    if m:
        return int(m.group(1))
    # If we see "1 error during collection" or similar, treat as a large failure.
    if "error" in test_output.lower() and "passed" not in test_output.lower():
        return 999
    return 0


def _verify_patch(instance: Instance, patch: str) -> tuple[bool, int, str]:
    """Apply patch to a fresh checkout of `instance` and run the tests."""
    if not patch.strip():
        return False, 999, "empty patch"
    with tempfile.TemporaryDirectory(prefix=f"verify-{instance.instance_id}-") as td:
        repo = Path(td) / "repo"
        _materialize(instance, repo)
        applied, _apply_log = _apply_patch(patch, repo)
        if not applied:
            return False, 999, "patch did not apply"
        proc = subprocess.run(
            instance.test_command,
            shell=True, cwd=repo, text=True, timeout=120,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        out = proc.stdout
        if proc.returncode == 0:
            return True, 0, out[-1000:]
        return False, _count_failing(out), out[-1000:]


def rank_candidates(instance: Instance, candidates: list[Path], config: dict) -> CandidateScore | None:
    """Score every candidate trajectory's patch and return the winner."""
    scored: list[CandidateScore] = []
    for traj_dir in candidates:
        traj_path = traj_dir / f"{instance.instance_id}.traj.json"
        if not traj_path.exists():
            continue
        traj = json.loads(traj_path.read_text())
        patch = (traj.get("info") or {}).get("submission") or ""
        ok, failing, _ = _verify_patch(instance, patch)
        scored.append(CandidateScore(
            trajectory_dir=traj_dir,
            patch=patch,
            tests_pass=ok,
            failing_tests=failing,
            patch_size=len(patch),
        ))
    if not scored:
        return None
    # Rank: tests_pass desc, failing_tests asc, patch_size asc
    scored.sort(key=lambda s: (not s.tests_pass, s.failing_tests, s.patch_size))
    return scored[0]


def run_instance_bestofn(
    instance: Instance,
    config: dict,
    output_dir: Path,
    n: int = 3,
) -> dict:
    """Run `n` independent rollouts; copy the winner's outputs to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_dirs: list[Path] = []
    for k in range(n):
        rk = output_dir / f"rollout_{k}"
        rk.mkdir(parents=True, exist_ok=True)
        # Vary RNG / sampling: bump temperature for samples k>0; deterministic at k=0
        config_k = json.loads(json.dumps(config))
        if k > 0:
            mk = config_k.setdefault("model", {}).setdefault("model_kwargs", {})
            mk["temperature"] = max(0.2, mk.get("temperature", 0.0) + 0.2 * k)
            mk["seed"] = 1000 + k
        run_instance(instance, config_k, rk)
        rollout_dirs.append(rk / instance.instance_id)

    winner = rank_candidates(instance, rollout_dirs, config)
    verdict = {"instance_id": instance.instance_id, "rollouts": n}
    if winner is None:
        verdict["resolved"] = False
        return verdict

    # Promote winner's outputs to output_dir/<instance>/
    winner_src = winner.trajectory_dir
    dest = output_dir / instance.instance_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(winner_src, dest)

    verdict.update({
        "resolved": winner.tests_pass,
        "patch_applied": winner.tests_pass or winner.failing_tests < 999,
        "tests_passed_after": winner.tests_pass,
        "patch": winner.patch,
        "winner_rollout": winner.trajectory_dir.parent.name,
        "all_rollouts": [r.parent.name for r in rollout_dirs],
    })
    (dest / "verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def run_all_bestofn(
    fixtures_dir: Path,
    config: dict,
    output_dir: Path,
    n: int = 3,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = discover_fixtures(fixtures_dir)
    preds: dict = {}
    resolved_ids: list[str] = []
    verdicts: list[dict] = []
    import time
    t0 = time.time()
    for inst in instances:
        v = run_instance_bestofn(inst, config, output_dir, n=n)
        verdicts.append(v)
        preds[inst.instance_id] = {
            "instance_id": inst.instance_id,
            "model_patch": v.get("patch") or "",
            "model_name_or_path": config.get("model", {}).get("model_name", "unknown"),
        }
        if v.get("resolved"):
            resolved_ids.append(inst.instance_id)
        (output_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    report = {
        "resolved_ids": resolved_ids,
        "n_total": len(instances),
        "n_resolved": len(resolved_ids),
        "elapsed_seconds": time.time() - t0,
        "n_rollouts_per_instance": n,
        "verdicts": verdicts,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
