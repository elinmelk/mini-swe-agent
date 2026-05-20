"""`projectk-mini` — Docker-free mini-benchmark runner.

Example:
    projectk-mini --output runs/mini1
    projectk-mini --output runs/mini1 -c src/minisweagent/config/projectk/nvidia.yaml

Outputs the same SWE-Bench-style files (`preds.json`, per-instance
`*.traj.json`) plus a local `report.json` with `resolved_ids`, so
`projectk-report runs/mini1 --report runs/mini1/report.json` works end-to-end
without Docker.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.projectk.cli_report import print_report
from minisweagent.projectk.minibench.runner import run_all
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

DEFAULT_FIXTURES = Path(__file__).parent / "minibench" / "fixtures"
DEFAULT_CONFIG = builtin_config_dir / "projectk" / "nvidia.yaml"


@app.command()
def main(
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for traj + report"),
    fixtures_dir: Path = typer.Option(DEFAULT_FIXTURES, "--fixtures", help="Directory of fixtures"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG)],
        "-c",
        "--config",
        help="One or more config files / overrides (recursively merged)",
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Override model name"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Override agent class"),
    cost_limit: float | None = typer.Option(None, "--cost-limit"),
    step_limit: int | None = typer.Option(None, "--step-limit"),
) -> None:
    configs = [get_config_from_spec(spec) for spec in config_spec]
    overrides: dict = {}
    if model:
        overrides.setdefault("model", {})["model_name"] = model
    if agent_class:
        overrides.setdefault("agent", {})["agent_class"] = agent_class
    if cost_limit is not None:
        overrides.setdefault("agent", {})["cost_limit"] = cost_limit
    if step_limit is not None:
        overrides.setdefault("agent", {})["step_limit"] = step_limit
    if overrides:
        configs.append(overrides)
    config = recursive_merge(*configs)

    console.rule(f"[bold cyan]projectk-mini — {fixtures_dir}")
    report = run_all(fixtures_dir, config, output)
    console.print(
        f"[bold green]Done.[/] Resolved {report['n_resolved']}/{report['n_total']} "
        f"in {report['elapsed_seconds']:.1f}s"
    )

    console.rule("[bold]Project K report")
    print_report(output, report=output / "report.json", json_out=output / "projectk_report.json")


if __name__ == "__main__":
    app()
