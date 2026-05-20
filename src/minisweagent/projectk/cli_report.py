"""`projectk-report` — print Project K metrics + failure analysis for a run dir."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from minisweagent.projectk.failure_analysis import analyze_directory
from minisweagent.projectk.metrics import compute_metrics, load_resolved_from_report

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()


def print_report(output_dir: Path, report: Path | None = None, json_out: Path | None = None) -> dict:
    resolved_ids: set[str] = set()
    if report is not None and Path(report).exists():
        resolved_ids = load_resolved_from_report(report)

    metrics = compute_metrics(output_dir, resolved_instances=resolved_ids)
    failures = analyze_directory(output_dir, resolved_ids=resolved_ids)

    summary_table = Table(title="Project K — run metrics", title_style="bold")
    summary_table.add_column("metric", style="cyan")
    summary_table.add_column("value", style="green")
    for k, v in metrics.to_dict().items():
        if k == "per_instance":
            continue
        if k == "exit_status_counts":
            summary_table.add_row(k, ", ".join(f"{kk}={vv}" for kk, vv in v.items()) or "-")
        elif isinstance(v, float):
            summary_table.add_row(k, f"{v:.3f}")
        else:
            summary_table.add_row(k, str(v))
    console.print(summary_table)

    fail_table = Table(title="Failure-mode taxonomy", title_style="bold")
    fail_table.add_column("label", style="cyan")
    fail_table.add_column("count", style="green")
    for label, count in sorted(failures["summary"].items(), key=lambda x: -x[1]):
        fail_table.add_row(label, str(count))
    console.print(fail_table)

    full = {"metrics": metrics.to_dict(), "failures": failures}
    if json_out:
        Path(json_out).write_text(json.dumps(full, indent=2))
        console.print(f"[bold green]Wrote[/bold green] {json_out}")
    return full


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Directory holding *.traj.json + preds.json"),
    report: Path | None = typer.Option(None, "--report", help="Optional SWE-Bench evaluator report.json"),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the full report as JSON to this path"),
) -> None:
    """Aggregate trajectories from `output_dir` into a Project-K-style report."""
    print_report(output_dir, report=report, json_out=json_out)


if __name__ == "__main__":
    app()
