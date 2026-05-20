"""Docker-free mini-benchmark runner.

Walks a fixtures directory, runs the configured agent against each instance in
an isolated tmpdir, and writes SWE-Bench-style outputs:

    output_dir/
        <instance_id>/<instance_id>.traj.json
        preds.json
        report.json                 # local equivalent of SWE-Bench's evaluator
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from minisweagent.agents import get_agent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.utils.log import logger


@dataclass
class Instance:
    instance_id: str
    problem_statement: str
    test_command: str
    repo_dir: Path
    fixture_dir: Path
    setup_commands: list[str]

    @classmethod
    def from_dir(cls, fixture_dir: Path) -> "Instance":
        meta = yaml.safe_load((fixture_dir / "instance.yaml").read_text())
        return cls(
            instance_id=meta["instance_id"],
            problem_statement=meta["problem_statement"],
            test_command=meta.get("test_command", "pytest -q"),
            repo_dir=fixture_dir / "repo",
            fixture_dir=fixture_dir,
            setup_commands=list(meta.get("setup_commands") or []),
        )


def _run(cmd: str, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )


_DEFAULT_GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\n.cache/\n"


def _materialize(instance: Instance, dest: Path) -> None:
    shutil.copytree(instance.repo_dir, dest, dirs_exist_ok=True)
    gi = dest / ".gitignore"
    if not gi.exists():
        gi.write_text(_DEFAULT_GITIGNORE)
    _run("git init -q && git add -A && git -c user.email=mini@example.com -c user.name=mini "
         "commit -q -m initial", cwd=dest)
    for setup in instance.setup_commands:
        result = _run(setup, cwd=dest, timeout=300)
        if result.returncode != 0:
            logger.warning(f"[{instance.instance_id}] setup failed: {setup}\n{result.stdout}")


def _tests_pass(test_command: str, cwd: Path) -> tuple[bool, str]:
    try:
        result = _run(test_command, cwd=cwd, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return result.returncode == 0, result.stdout[-2000:]


_NOISY_PATH_SEGMENTS = ("__pycache__", ".pytest_cache", ".cache", ".mypy_cache", ".ruff_cache")


def _strip_noisy_diffs(patch: str) -> str:
    """Remove file-diffs that target __pycache__ etc., which never apply on a fresh checkout."""
    out: list[str] = []
    keep = True
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            parts = line.split()
            target = parts[-1][2:] if len(parts) >= 4 else ""
            segments = target.split("/")
            keep = not any(seg in _NOISY_PATH_SEGMENTS for seg in segments)
        if keep:
            out.append(line)
    return "".join(out)


def _apply_patch(patch: str, cwd: Path) -> tuple[bool, str]:
    patch = _strip_noisy_diffs(patch)
    if not patch.strip():
        return False, "EMPTY_PATCH"
    proc = subprocess.run(
        "git apply --whitespace=nowarn -",
        input=patch, shell=True, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.returncode == 0, proc.stdout


def _extract_patch(traj: dict) -> str:
    """Pull the final submitted patch out of an agent trajectory."""
    info = traj.get("info") or {}
    submission = info.get("submission") or ""
    return submission


def _build_agent(config: dict, env: LocalEnvironment):
    model = get_model(config=config.get("model", {}))
    return get_agent(model, env, config.get("agent", {}), default_type="default")


def run_instance(instance: Instance, config: dict, output_dir: Path) -> dict[str, Any]:
    inst_out = output_dir / instance.instance_id
    inst_out.mkdir(parents=True, exist_ok=True)
    traj_path = inst_out / f"{instance.instance_id}.traj.json"

    with tempfile.TemporaryDirectory(prefix=f"projk-{instance.instance_id}-") as td:
        repo = Path(td) / "repo"
        _materialize(instance, repo)

        pre_pass, pre_out = _tests_pass(instance.test_command, repo)
        if pre_pass:
            logger.warning(f"[{instance.instance_id}] tests pass BEFORE agent — fixture is mis-written?")

        env_config = dict(config.get("environment") or {})
        env_config["cwd"] = str(repo)
        env = LocalEnvironment(**env_config)

        agent = _build_agent(config, env)
        info: dict[str, Any] = {}
        try:
            info = agent.run(instance.problem_statement)
        except Exception as e:
            logger.error(f"[{instance.instance_id}] agent crashed: {e}", exc_info=True)
            info = {"exit_status": type(e).__name__, "submission": "", "traceback": traceback.format_exc()}
        agent.save(traj_path, {"instance_id": instance.instance_id, "info": info})

        # Verify by re-applying the patch against a fresh checkout
        patch = _extract_patch(json.loads(traj_path.read_text()))
        verdict: dict[str, Any] = {
            "instance_id": instance.instance_id,
            "tests_passed_before": pre_pass,
            "patch_applied": False,
            "tests_passed_after": False,
            "test_output_tail": "",
        }
        if patch.strip():
            verify = Path(td) / "verify"
            _materialize(instance, verify)
            applied, apply_log = _apply_patch(patch, verify)
            verdict["patch_applied"] = applied
            verdict["apply_log"] = apply_log[-1000:]
            if applied:
                post_pass, post_out = _tests_pass(instance.test_command, verify)
                verdict["tests_passed_after"] = post_pass
                verdict["test_output_tail"] = post_out

        verdict["resolved"] = bool(
            verdict["patch_applied"]
            and verdict["tests_passed_after"]
            and not verdict["tests_passed_before"]
        )
        verdict["patch"] = patch
        (inst_out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        return verdict


def discover_fixtures(fixtures_dir: Path) -> list[Instance]:
    return [
        Instance.from_dir(p)
        for p in sorted(fixtures_dir.iterdir())
        if p.is_dir() and (p / "instance.yaml").exists()
    ]


def run_all(fixtures_dir: Path, config: dict, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = discover_fixtures(fixtures_dir)
    preds: dict[str, Any] = {}
    resolved_ids: list[str] = []
    verdicts: list[dict[str, Any]] = []
    started = time.time()
    for inst in instances:
        logger.info(f"=== {inst.instance_id} ===")
        verdict = run_instance(inst, config, output_dir)
        verdicts.append(verdict)
        preds[inst.instance_id] = {
            "instance_id": inst.instance_id,
            "model_patch": verdict.get("patch") or "",
            "model_name_or_path": config.get("model", {}).get("model_name", "unknown"),
        }
        if verdict.get("resolved"):
            resolved_ids.append(inst.instance_id)
        (output_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    report = {
        "resolved_ids": resolved_ids,
        "n_total": len(instances),
        "n_resolved": len(resolved_ids),
        "elapsed_seconds": time.time() - started,
        "verdicts": verdicts,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
