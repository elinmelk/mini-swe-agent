"""`projectk-run` — convenience wrapper around the SWE-Bench batch runner with
Project K defaults (NVIDIA NIM config + curated 20-instance slice).

Equivalent to:

    python -m minisweagent.run.benchmarks.swebench \
        --subset lite --split dev \
        --slice 0:20 \
        -c src/minisweagent/config/projectk/swebench_lite_nvidia.yaml \
        -o runs/projectk_$(date +%s) \
        -m nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct

…with metrics + failure analysis run automatically afterward.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from minisweagent.config import builtin_config_dir
from minisweagent.projectk.cli_report import print_report

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()


@app.command()
def main(
    output: Path = typer.Option(..., "--output", "-o", help="Run output directory"),
    model: str = typer.Option(
        "nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct",
        "--model",
        "-m",
        help="LLM model id (litellm-format)",
    ),
    subset: str = typer.Option("lite", "--subset"),
    split: str = typer.Option("dev", "--split"),
    slice_spec: str = typer.Option("0:20", "--slice", help="Slice of the dataset (default first 20)"),
    workers: int = typer.Option(1, "--workers", "-w"),
    config: Path = typer.Option(
        builtin_config_dir / "projectk" / "swebench_lite_nvidia.yaml",
        "--config",
        "-c",
    ),
    skip_run: bool = typer.Option(False, "--skip-run", help="Just regenerate metrics/failure report"),
    report_json: Path | None = typer.Option(None, "--report", help="SWE-Bench evaluator report.json"),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if not skip_run:
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
            str(output),
            "-c",
            str(config),
            "-m",
            model,
        ]
        console.print(f"[bold green]$[/] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            console.print(f"[bold red]Batch run exited with code {result.returncode}[/]")

    console.rule("[bold]Project K report")
    print_report(output, report=report_json, json_out=output / "projectk_report.json")


if __name__ == "__main__":
    app()
