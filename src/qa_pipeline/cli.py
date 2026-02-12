
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .pipeline import QAPipeline
from .config import QAConfig

console = Console()


@click.group()
def cli():

    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to raw dataset (CSV or Parquet)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file (optional)",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run identifier (auto-generated if not provided)",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(),
    default="artifacts",
    help="Base directory for artifacts",
)
@click.option(
    "--soft-fail",
    is_flag=True,
    help="Exit with code 0 even if QA checks fail",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Skip plot generation",
)
def run(
    input_path: str,
    config_path: str | None,
    run_id: str | None,
    artifacts_dir: str,
    soft_fail: bool,
    no_plots: bool,
):

    console.print("\n[bold blue]🔍 QA Pipeline Starting...[/bold blue]\n")

    try:
        if config_path:
            console.print(f"Loading config from: [cyan]{config_path}[/cyan]")
            config = QAConfig.from_yaml(config_path)
        else:
            console.print("Using default configuration")
            config = QAConfig.load()
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[dim]Timezone policy: {config.timezone_policy}[/dim]")
    console.print(f"[dim]DST policy: {config.dst_policy}[/dim]")
    console.print(f"[dim]LLM: {config.llm_provider}/{config.llm_model}[/dim]\n")

    try:
        pipeline = QAPipeline(
            config=config,
            artifacts_dir=Path(artifacts_dir),
            run_id=run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing pipeline:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[bold green]Running pipeline on:[/bold green] {input_path}\n")

    try:
        with console.status("[bold yellow]Processing...", spinner="dots"):
            result = pipeline.run(input_path)

        if not result.get("success"):
            console.print(
                f"\n[bold red]Pipeline failed:[/bold red] {result.get('error')}\n"
            )
            sys.exit(1)

        console.print("\n[bold green]✅ Pipeline completed![/bold green]\n")

        table = Table(title="QA Pipeline Results")
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Run ID", result["run_id"])
        table.add_row(
            "Training Ready",
            "✅ YES" if result["training_ready"] else "❌ NO",
        )
        table.add_row("Clean Dataset", result.get("clean_dataset_path", "N/A"))
        table.add_row("QA Report", result["qa_report_path"])
        table.add_row("QA Results (JSON)", result["qa_results_path"])
        table.add_row("Artifacts Directory", result["artifacts_dir"])

        console.print(table)

        if result.get("pipeline_output"):
            console.print(f"\n[bold]Pipeline Summary:[/bold]")
            console.print(result["pipeline_output"])

        if not result["training_ready"] and not soft_fail:
            console.print(
                "\n[bold yellow]⚠️  Dataset not ready for training. Check QA report for details.[/bold yellow]\n"
            )
            sys.exit(1)
        else:
            console.print("\n[bold green]✨ All done![/bold green]\n")
            sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[bold red]Pipeline interrupted by user[/bold red]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.yaml",
    help="Output path for config file",
)
def init_config(output: str):

    output_path = Path(output)

    if output_path.exists():
        console.print(
            f"[bold yellow]Config file already exists:[/bold yellow] {output}"
        )
        if not click.confirm("Overwrite?"):
            console.print("Aborted.")
            return

    try:
        config = QAConfig()
        config.to_yaml(output_path)
        console.print(f"[bold green]✅ Config file created:[/bold green] {output}")
    except Exception as e:
        console.print(f"[bold red]Error creating config:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def version():

    from . import __version__

    console.print(f"QA Pipeline version: [bold cyan]{__version__}[/bold cyan]")


def main():

    cli()


if __name__ == "__main__":
    main()
