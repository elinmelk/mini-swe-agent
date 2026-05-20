"""Aggregate per-instance trajectories into Project K metrics.

We compute, across an output directory of `*.traj.json` files plus the
`preds.json` and (optionally) the SWE-Bench `report.json`:

  - resolve_rate         = resolved / submitted
  - submission_rate      = submitted (non-empty patch) / attempted
  - mean_tool_calls      = mean number of agent steps per attempt
  - mean_cost_usd        = mean computed cost (0 if cost_tracking is off)
  - mean_input_tokens    = mean prompt tokens per attempt (when available)
  - mean_output_tokens   = mean completion tokens per attempt
  - mean_latency_seconds = wall-clock duration per attempt (from timestamps)
  - exit_status_counts   = histogram of agent exit_status values
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Metrics:
    n_attempts: int = 0
    n_submitted: int = 0
    n_resolved: int = 0
    resolve_rate: float = 0.0
    submission_rate: float = 0.0
    mean_tool_calls: float = 0.0
    mean_cost_usd: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_latency_seconds: float = 0.0
    exit_status_counts: dict[str, int] = field(default_factory=dict)
    per_instance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_mean(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else 0.0


def _trajectory_token_counts(traj: dict) -> tuple[int, int]:
    """Sum input/output tokens across all model responses in a trajectory."""
    in_tokens = out_tokens = 0
    for msg in traj.get("messages", []):
        usage = (msg.get("extra") or {}).get("response", {}).get("usage") or {}
        in_tokens += int(usage.get("prompt_tokens") or 0)
        out_tokens += int(usage.get("completion_tokens") or 0)
    return in_tokens, out_tokens


def _trajectory_latency(traj: dict) -> float:
    ts = [
        (msg.get("extra") or {}).get("timestamp")
        for msg in traj.get("messages", [])
        if (msg.get("extra") or {}).get("timestamp") is not None
    ]
    if len(ts) < 2:
        return 0.0
    return float(max(ts) - min(ts))


def compute_metrics(
    output_dir: str | Path,
    resolved_instances: set[str] | None = None,
) -> Metrics:
    """Walk an output directory and aggregate metrics.

    ``resolved_instances`` should be the set of instance_ids that the official
    SWE-Bench evaluator marked as resolved. When not provided, ``resolve_rate``
    is left at 0 and ``n_resolved`` records nothing (we still report submission
    rate, which is verifiable locally).
    """
    output_dir = Path(output_dir)
    resolved_instances = resolved_instances or set()
    preds_path = output_dir / "preds.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}

    tool_calls: list[int] = []
    costs: list[float] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    latencies: list[float] = []
    exit_counts: dict[str, int] = {}
    per_inst: list[dict[str, Any]] = []
    n_submitted = 0
    n_resolved = 0

    for inst_dir in sorted(output_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        traj_path = inst_dir / f"{inst_dir.name}.traj.json"
        if not traj_path.exists():
            continue
        traj = json.loads(traj_path.read_text())
        info = traj.get("info", {})
        instance_id = traj.get("instance_id") or inst_dir.name
        exit_status = info.get("exit_status", "Unknown")
        exit_counts[exit_status] = exit_counts.get(exit_status, 0) + 1

        api_calls = int((info.get("model_stats") or {}).get("api_calls") or 0)
        cost = float((info.get("model_stats") or {}).get("instance_cost") or 0.0)
        i_tok, o_tok = _trajectory_token_counts(traj)
        latency = _trajectory_latency(traj)

        submission = (preds.get(instance_id) or {}).get("model_patch") or info.get("submission") or ""
        is_submitted = bool(submission.strip())
        is_resolved = instance_id in resolved_instances
        if is_submitted:
            n_submitted += 1
        if is_resolved:
            n_resolved += 1

        tool_calls.append(api_calls)
        costs.append(cost)
        in_tokens.append(i_tok)
        out_tokens.append(o_tok)
        latencies.append(latency)

        per_inst.append({
            "instance_id": instance_id,
            "exit_status": exit_status,
            "submitted": is_submitted,
            "resolved": is_resolved,
            "tool_calls": api_calls,
            "cost_usd": cost,
            "input_tokens": i_tok,
            "output_tokens": o_tok,
            "latency_seconds": latency,
        })

    n_attempts = len(per_inst)
    return Metrics(
        n_attempts=n_attempts,
        n_submitted=n_submitted,
        n_resolved=n_resolved,
        resolve_rate=(n_resolved / n_attempts) if n_attempts else 0.0,
        submission_rate=(n_submitted / n_attempts) if n_attempts else 0.0,
        mean_tool_calls=_safe_mean(tool_calls),
        mean_cost_usd=_safe_mean(costs),
        mean_input_tokens=_safe_mean(in_tokens),
        mean_output_tokens=_safe_mean(out_tokens),
        mean_latency_seconds=_safe_mean(latencies),
        exit_status_counts=exit_counts,
        per_instance=per_inst,
    )


def load_resolved_from_report(report_path: str | Path) -> set[str]:
    """Parse the SWE-Bench evaluator's report.json and return resolved IDs."""
    data = json.loads(Path(report_path).read_text())
    if "resolved_ids" in data:
        return set(data["resolved_ids"])
    if "resolved" in data:
        return set(data["resolved"])
    out: set[str] = set()
    for instance_id, val in data.items():
        if isinstance(val, dict) and val.get("resolved"):
            out.add(instance_id)
    return out
