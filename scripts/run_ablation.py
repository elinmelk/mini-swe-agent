"""Run the Project K ablation grid: 4 conditions × N fixtures.

Conditions:
  baseline    — DefaultAgent, no enhancements
  planner     — Plan-Execute decomposition
  scratchpad  — Persistent memory file across steps
  retrieval   — Symbol-graph retrieval injected into the first prompt

For each (condition, fixture) we run projectk-mini in its own output dir and
aggregate metrics + verdicts into:

  runs/ablation/results.csv        # one row per (condition, fixture)
  runs/ablation/summary.json       # per-condition aggregates
"""

from __future__ import annotations

import argparse
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
    "baseline":   CFG_DIR / "ollama.yaml",
    "planner":    CFG_DIR / "ollama_planner.yaml",
    "scratchpad": CFG_DIR / "ollama_scratchpad.yaml",
    "retrieval":  CFG_DIR / "ollama_retrieval.yaml",
}


def _run(model: str, config: Path, fixtures: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "minisweagent.projectk.cli_mini",
        "-o", str(output), "-c", str(config), "-m", model,
        "--fixtures", str(fixtures),
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Project K ablation grid")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="ollama_chat/qwen2.5-coder:14b")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES_DIR,
                    help="Fixtures directory (default: all bundled)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for cond in args.conditions:
        cfg = CONDITIONS[cond]
        cond_dir = args.output / cond
        print(f"\n=== condition: {cond}  (config={cfg.name}) ===")
        _run(args.model, cfg, args.fixtures, cond_dir)

        report_path = cond_dir / "report.json"
        resolved = set()
        if report_path.exists():
            resolved = load_resolved_from_report(report_path)
        m = compute_metrics(cond_dir, resolved_instances=resolved)
        summary[cond] = {
            "n_attempts": m.n_attempts,
            "n_resolved": m.n_resolved,
            "resolve_rate": m.resolve_rate,
            "submission_rate": m.submission_rate,
            "mean_tool_calls": m.mean_tool_calls,
            "mean_input_tokens": m.mean_input_tokens,
            "mean_output_tokens": m.mean_output_tokens,
            "mean_latency_seconds": m.mean_latency_seconds,
            "mean_cost_usd": m.mean_cost_usd,
            "exit_status_counts": m.exit_status_counts,
        }
        for p in m.per_instance:
            rows.append({"condition": cond, **p})

    # Persist results
    csv_path = args.output / "results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print headline table
    print("\n" + "=" * 72)
    print("CONDITION       RES%  SUB%  STEPS  IN-TOK  OUT-TOK  LAT(s)  COST($)")
    print("-" * 72)
    for cond, s in summary.items():
        print(f"{cond:<14} {s['resolve_rate']*100:>4.0f}% {s['submission_rate']*100:>4.0f}% "
              f"{s['mean_tool_calls']:>6.1f} {s['mean_input_tokens']:>7.0f} "
              f"{s['mean_output_tokens']:>8.0f} {s['mean_latency_seconds']:>6.1f} "
              f"{s['mean_cost_usd']:>6.4f}")
    print("=" * 72)
    print(f"\nWrote:")
    print(f"  {csv_path}")
    print(f"  {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
