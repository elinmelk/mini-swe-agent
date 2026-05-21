"""Expanded ablation: baseline + 3 scaffolds + 4 optimizations on the same model.

Conditions:
  baseline          DefaultAgent
  planner           plan-execute (replan_every=0)
  scratchpad        persistent memory
  retrieval         symbol-graph
  bestofn_baseline  Best-of-N=3 over baseline
  reflexion         retry-on-failure with self-critique
  retrieval_hybrid  symbol + BM25 fused via RRF
  planner_dynamic   planner with on-demand replanning (>=3 consecutive failures)

Writes:
  runs/opt_ablation/<cond>/...
  runs/opt_ablation/results.csv
  runs/opt_ablation/summary.json
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from minisweagent.projectk.metrics import compute_metrics, load_resolved_from_report

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "src" / "minisweagent" / "config" / "projectk"
FIXTURES_DIR = ROOT / "src" / "minisweagent" / "projectk" / "minibench" / "fixtures"


CONDITIONS = {
    # Original 4
    "baseline":         {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama.yaml"},
    "planner":          {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama_planner.yaml"},
    "scratchpad":       {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama_scratchpad.yaml"},
    "retrieval":        {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama_retrieval.yaml"},
    # New optimization techniques
    "bestofn_baseline": {"cli": "minisweagent.projectk.cli_bestofn",     "cfg": "ollama.yaml",            "extra": ["-n", "3"]},
    "reflexion":        {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama_reflexion.yaml"},
    "retrieval_hybrid": {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama_retrieval_hybrid.yaml"},
    "planner_dynamic":  {"cli": "minisweagent.projectk.cli_mini",        "cfg": "ollama_planner_dynamic.yaml"},
}


def _run(cli: str, cfg: Path, fixtures: Path, output: Path, extra: list[str], model: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", cli,
           "-o", str(output), "-c", str(cfg),
           "-m", model, "--fixtures", str(fixtures)] + extra
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="ollama_chat/qwen2.5-coder:14b")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES_DIR)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for cond in args.conditions:
        meta = CONDITIONS[cond]
        cfg = CFG_DIR / meta["cfg"]
        extra = meta.get("extra", [])
        cond_dir = args.output / cond
        print(f"\n=== {cond} ===")
        _run(meta["cli"], cfg, args.fixtures, cond_dir, extra, args.model)

        report_path = cond_dir / "report.json"
        resolved = set()
        if report_path.exists():
            resolved = load_resolved_from_report(report_path)
        m = compute_metrics(cond_dir, resolved_instances=resolved)
        summary[cond] = {
            "n_attempts": m.n_attempts, "n_resolved": m.n_resolved,
            "resolve_rate": m.resolve_rate, "submission_rate": m.submission_rate,
            "mean_tool_calls": m.mean_tool_calls,
            "mean_input_tokens": m.mean_input_tokens,
            "mean_output_tokens": m.mean_output_tokens,
            "mean_latency_seconds": m.mean_latency_seconds,
            "mean_cost_usd": m.mean_cost_usd,
            "exit_status_counts": m.exit_status_counts,
        }
        for p in m.per_instance:
            rows.append({"condition": cond, **p})

    csv_path = args.output / "results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("CONDITION          RES%  SUB%  STEPS  IN-TOK  OUT-TOK  LAT(s)")
    print("-" * 80)
    for cond, s in summary.items():
        print(f"{cond:<18} {s['resolve_rate']*100:>4.0f}% {s['submission_rate']*100:>4.0f}% "
              f"{s['mean_tool_calls']:>6.1f} {s['mean_input_tokens']:>7.0f} "
              f"{s['mean_output_tokens']:>8.0f} {s['mean_latency_seconds']:>6.1f}")
    print("=" * 80)
    print(f"\nWrote {csv_path}\nWrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
