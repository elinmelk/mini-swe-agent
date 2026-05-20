"""`projectk-mini-compare` — Docker-free comparison of N models on the same
mini-benchmark slice. Each model can use its own config (so Ollama uses
api_base=localhost, Groq uses GROQ_API_KEY, etc.).

Example:
    projectk-mini-compare -o runs/cmp \
        --pair "ollama_chat/qwen2.5-coder:14b=src/minisweagent/config/projectk/ollama.yaml" \
        --pair "groq/llama-3.3-70b-versatile=src/minisweagent/config/projectk/groq.yaml"
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from minisweagent.projectk.metrics import compute_metrics, load_resolved_from_report

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()


def _slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def _run_one(model: str, config: Path, fixtures: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "minisweagent.projectk.cli_mini",
        "-o",
        str(output),
        "-c",
        str(config),
        "-m",
        model,
        "--fixtures",
        str(fixtures),
    ]
    console.print(f"[bold green]$[/] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print(f"[bold red]Run for {model} exited with {result.returncode}[/]")


@app.command()
def main(
    output: Path = typer.Option(..., "-o", "--output", help="Output root directory"),
    pairs: list[str] = typer.Option(
        ...,
        "--pair",
        help='Model + config pair: "MODEL_NAME=PATH/TO/config.yaml". Repeat for each model.',
    ),
    fixtures: Path = typer.Option(
        Path(__file__).parent / "minibench" / "fixtures",
        "--fixtures",
        help="Fixtures directory (defaults to bundled 5 toy bugs)",
    ),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, Path]] = []
    for raw in pairs:
        parts = raw.split("=")
        if len(parts) == 2:
            label, model, config = parts[0], parts[0], parts[1]
        elif len(parts) >= 3:
            label, model, config = parts[0], parts[1], "=".join(parts[2:])
        else:
            console.print(f"[bold red]Bad --pair (need MODEL=CONFIG or LABEL=MODEL=CONFIG): {raw}[/]")
            raise typer.Exit(2)
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[bold red]Config not found: {config_path}[/]")
            raise typer.Exit(2)
        out_dir = output / _slug(label)
        out_dir.mkdir(parents=True, exist_ok=True)
        console.rule(f"[bold cyan]{label}  (model={model})")
        _run_one(model, config_path, fixtures, out_dir)
        runs.append((label, out_dir))

    table = Table(title="Project K — open-weight model comparison")
    table.add_column("model", style="cyan")
    table.add_column("resolve %", style="green")
    table.add_column("submit %", style="green")
    table.add_column("mean steps", style="yellow")
    table.add_column("mean in-tok", style="yellow")
    table.add_column("mean out-tok", style="yellow")
    table.add_column("mean lat s", style="yellow")
    table.add_column("mean cost $", style="yellow")

    combined: dict[str, dict] = {}
    for model, run_dir in runs:
        resolved = set()
        report = run_dir / "report.json"
        if report.exists():
            resolved = load_resolved_from_report(report)
        m = compute_metrics(run_dir, resolved_instances=resolved).to_dict()
        combined[model] = m
        table.add_row(
            model,
            f"{m['resolve_rate'] * 100:.1f}",
            f"{m['submission_rate'] * 100:.1f}",
            f"{m['mean_tool_calls']:.1f}",
            f"{m['mean_input_tokens']:.0f}",
            f"{m['mean_output_tokens']:.0f}",
            f"{m['mean_latency_seconds']:.1f}",
            f"{m['mean_cost_usd']:.4f}",
        )

    console.print(table)
    (output / "comparison.json").write_text(json.dumps(combined, indent=2))
    console.print(f"[bold green]Wrote[/bold green] {output / 'comparison.json'}")


if __name__ == "__main__":
    app()
