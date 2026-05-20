"""`projectk-compare` — run the same benchmark slice across multiple models.

Example:
    projectk-compare \
        --output runs/compare \
        --models nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct \
        --models nvidia_nim/meta/llama-3.3-70b-instruct \
        --slice 0:5

Each model is run via the existing `minisweagent.run.benchmarks.swebench` batch
script, isolated under its own subdirectory, and a comparison table is printed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from minisweagent.config import builtin_config_dir
from minisweagent.projectk.metrics import compute_metrics, load_resolved_from_report

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


@app.command()
def main(
    models: list[str] = typer.Option(..., "--models", "-m", help="Models to compare (repeat the flag)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output root directory"),
    config: Path = typer.Option(
        builtin_config_dir / "projectk" / "swebench_lite_nvidia.yaml",
        "--config",
        "-c",
        help="Benchmark config to use (one config, multiple models)",
    ),
    subset: str = typer.Option("lite", "--subset", help="SWE-Bench subset"),
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("0:5", "--slice", help="Slice (e.g. 0:5)"),
    workers: int = typer.Option(1, "--workers", "-w"),
    reports_dir: Path | None = typer.Option(
        None,
        "--reports-dir",
        help="Optional directory containing per-model SWE-Bench evaluator report.json "
             "(named <model-slug>.report.json)",
    ),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    per_model_metrics: dict[str, dict] = {}

    for model in models:
        model_dir = output / _slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        console.rule(f"[bold cyan]Running {model}")
        cmd = [
            sys.executable,
            "-m",
            "minisweagent.run.benchmarks.swebench",
            "--subset",
            subset,
            "--split",
            split,
            "--slice",
            slice_spec,
            "--workers",
            str(workers),
            "-o",
            str(model_dir),
            "-c",
            str(config),
            "-m",
            model,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            console.print(f"[bold red]Run for {model} exited with {result.returncode}[/]")
        resolved_ids: set[str] = set()
        if reports_dir:
            candidate = Path(reports_dir) / f"{_slug(model)}.report.json"
            if candidate.exists():
                resolved_ids = load_resolved_from_report(candidate)
        metrics = compute_metrics(model_dir, resolved_instances=resolved_ids)
        per_model_metrics[model] = metrics.to_dict()
        (model_dir / "projectk_metrics.json").write_text(json.dumps(per_model_metrics[model], indent=2))

    table = Table(title="Project K — model comparison")
    table.add_column("model", style="cyan")
    table.add_column("resolve %", style="green")
    table.add_column("submit %", style="green")
    table.add_column("mean steps", style="yellow")
    table.add_column("mean cost $", style="yellow")
    table.add_column("mean in-tok", style="yellow")
    table.add_column("mean out-tok", style="yellow")
    table.add_column("mean lat s", style="yellow")
    for model, m in per_model_metrics.items():
        table.add_row(
            model,
            f"{m['resolve_rate'] * 100:.1f}",
            f"{m['submission_rate'] * 100:.1f}",
            f"{m['mean_tool_calls']:.1f}",
            f"{m['mean_cost_usd']:.4f}",
            f"{m['mean_input_tokens']:.0f}",
            f"{m['mean_output_tokens']:.0f}",
            f"{m['mean_latency_seconds']:.1f}",
        )
    console.print(table)

    (output / "comparison.json").write_text(json.dumps(per_model_metrics, indent=2))
    console.print(f"[bold green]Wrote[/bold green] {output / 'comparison.json'}")


if __name__ == "__main__":
    app()
