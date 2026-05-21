"""`projectk-mini-bestofn` — Best-of-N rollouts with test-based verifier.

Example:
    projectk-mini-bestofn -o runs/bestofn -n 3 \
        -c src/minisweagent/config/projectk/ollama.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.projectk.bestofn import run_all_bestofn
from minisweagent.projectk.cli_report import print_report
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

DEFAULT_FIXTURES = (
    Path(__file__).parent / "minibench" / "fixtures"
)
DEFAULT_CONFIG = builtin_config_dir / "projectk" / "ollama.yaml"


@app.command()
def main(
    output: Path = typer.Option(..., "--output", "-o"),
    n: int = typer.Option(3, "-n", "--rollouts", help="Number of independent rollouts per fixture"),
    fixtures_dir: Path = typer.Option(DEFAULT_FIXTURES, "--fixtures"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG)], "-c", "--config"),
    model: str | None = typer.Option(None, "-m", "--model"),
) -> None:
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if model:
        configs.append({"model": {"model_name": model}})
    config = recursive_merge(*configs)
    output.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]projectk-mini-bestofn (n={n})")
    report = run_all_bestofn(fixtures_dir, config, output, n=n)
    console.print(
        f"[bold green]Done.[/] Resolved {report['n_resolved']}/{report['n_total']} "
        f"in {report['elapsed_seconds']:.1f}s ({n} rollouts/fixture)"
    )
    console.rule("[bold]Project K report")
    print_report(output, report=output / "report.json",
                 json_out=output / "projectk_report.json")


if __name__ == "__main__":
    app()
