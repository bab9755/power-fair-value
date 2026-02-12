import sys
import time
from pathlib import Path

import click
from rich.console import Console

from .agent import InterpretationAgent
from .config import InterpretationConfig

console = Console()


def _display_streaming_progress(agent: InterpretationAgent) -> dict:
    result: dict = {}
    step_start = time.time()
    current_step = 0

    for event in agent.run_streaming():
        etype = event["event"]

        if etype == "tool_start":
            step_start = time.time()
            current_step = event["step"]
            label = event.get("label", event["tool"])
            detail = event.get("detail", "")
            step_str = f"[step {current_step}]"

            if detail:
                console.print(f"  {step_str} {label}  {detail}")
            else:
                console.print(f"  {step_str} {label}")

        elif etype == "tool_end":
            elapsed = round(time.time() - step_start, 1)
            detail = event.get("detail", "done")
            console.print(f"         {detail}  ({elapsed}s)")

        elif etype == "thinking":
            console.print(f"  Agent reasoning...")

        elif etype == "done":
            result = event.get("result", {})
            total = event.get("elapsed", 0)
            console.print(f"\n  Completed in {total}s")

        elif etype == "error":
            result = {
                "success": False,
                "error": event.get("error", "Unknown error"),
                "traceback": event.get("traceback", ""),
            }

    return result


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file (optional)",
)
@click.option(
    "--outputs-dir",
    type=str,
    default=None,
    help="Override the outputs directory",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def run(config_path: str | None, outputs_dir: str | None, verbose: bool):
    console.print("\nInterpretation Agent Starting...\n")

    try:
        if config_path:
            console.print(f"Loading config from: {config_path}")
            config = InterpretationConfig.from_yaml(config_path)
        else:
            config = InterpretationConfig.load()
            console.print("Using default configuration")
    except Exception as e:
        console.print(f"Error loading config: {e}")
        sys.exit(1)

    if outputs_dir:
        config.outputs_dir = outputs_dir
        config.da_to_curve_dir = str(Path(outputs_dir) / "da_to_curve")

    console.print(f"LLM: {config.llm_provider}/{config.llm_model}")
    console.print(f"Outputs dir: {config.outputs_dir}")
    console.print(f"Report output: {config.report_output}\n")

    try:
        agent = InterpretationAgent(config=config)
    except Exception as e:
        console.print(f"Error initializing agent: {e}")
        sys.exit(1)

    console.print("Running interpretation agent...\n")

    try:
        result = _display_streaming_progress(agent)

        if not result.get("success"):
            console.print(f"\nAgent failed: {result.get('error')}\n")
            if verbose and result.get("traceback"):
                console.print(result["traceback"])
            sys.exit(1)

        console.print("\nReport generated!\n")
        console.print(f"  Report: {result['report_path']}")
        console.print(f"  Artifacts loaded: {result['artifacts_loaded']}")
        console.print(f"  Images analyzed: {result['images_analyzed']}")
        console.print(f"  Web searches: {result.get('web_searches', 0)}")

        if verbose and result.get("agent_output"):
            console.print(f"\nAgent Output:")
            console.print(result["agent_output"])

        console.print("\nDone!\n")
        sys.exit(0)

    except KeyboardInterrupt:
        console.print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        console.print(f"\nUnexpected error: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file (optional)",
)
@click.option(
    "--outputs-dir",
    type=str,
    default=None,
    help="Override the outputs directory",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def chat(config_path: str | None, outputs_dir: str | None, verbose: bool):
    console.print("\nInterpretation Agent -- Chat Mode\n")

    try:
        if config_path:
            config = InterpretationConfig.from_yaml(config_path)
        else:
            config = InterpretationConfig.load()
    except Exception as e:
        console.print(f"Error loading config: {e}")
        sys.exit(1)

    if outputs_dir:
        config.outputs_dir = outputs_dir
        config.da_to_curve_dir = str(Path(outputs_dir) / "da_to_curve")

    try:
        agent = InterpretationAgent(config=config)
    except Exception as e:
        console.print(f"Error initializing agent: {e}")
        sys.exit(1)

    report_path = Path(config.report_output)
    if report_path.exists():
        console.print(f"Loading existing report from {report_path}")
        with open(report_path, "r") as f:
            agent.state["report_content"] = f.read()
        agent.state["report_path"] = str(report_path)
    else:
        console.print("No existing report found. Generating one first...\n")
        result = _display_streaming_progress(agent)

        if not result.get("success"):
            console.print(f"Failed to generate report: {result.get('error')}")
            sys.exit(1)

        console.print(f"\nReport generated: {result['report_path']}\n")

    console.print("Type your questions below. Type 'exit' or 'quit' to leave.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = console.input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("Goodbye!")
            break

        with console.status("Thinking...", spinner="dots"):
            response = agent.chat(user_input, history=history if history else None)

        console.print(f"\nAgent: {response}\n")

        history.append(("user", user_input))
        history.append(("assistant", response))


def main():
    cli()


if __name__ == "__main__":
    main()
